#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Online Exam Proctoring System - Main Application
================================================

This is the main application file that integrates all components of the
Online Exam Proctoring System. It provides a unified interface for monitoring
students during online examinations using computer vision techniques.

Author: Shohinur Pervez Shohan
GitHub: https://github.com/ShohanRony
"""

import cv2
import numpy as np
import argparse
import time
import os
from datetime import datetime

# Import component modules
from face_detector import get_face_detector, find_faces, draw_faces
from face_landmarks import get_landmark_model, detect_marks
from eye_tracker import eye_on_mask, find_eyeball_position, contouring, process_thresh
from head_pose_estimation import get_head_pose
from mouth_opening_detector import get_mouth_height
from face_spoofing import detect_spoofing

# Constants
FACE_DETECTION_INTERVAL = 5  # Process every 5th frame for performance
LOG_DIR = "logs"
ALERT_THRESHOLD = 3  # Number of suspicious activities before triggering alert


class OnlineProctor:
    """Main class for the Online Exam Proctoring System."""
    
    def __init__(self, video_source=0, log_dir=LOG_DIR):
        """Initialize the proctoring system.
        
        Args:
            video_source: Camera index or video file path
            log_dir: Directory to store logs and snapshots
        """
        self.video_source = video_source
        self.log_dir = log_dir
        self.cap = None
        self.frame_count = 0
        self.alerts = []
        self.suspicious_count = 0
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Initialize models
        self.face_model = get_face_detector()
        self.landmark_model = get_landmark_model()
        
        # Eye tracking parameters
        self.left_eye_idxs = [36, 37, 38, 39, 40, 41]
        self.right_eye_idxs = [42, 43, 44, 45, 46, 47]
        self.kernel = np.ones((9, 9), np.uint8)
        
        # Status tracking
        self.last_eye_direction = 0
        self.consecutive_eye_deviations = 0
        self.initial_mouth_height = None
        self.mouth_open_count = 0
        self.consecutive_no_face = 0
        self.consecutive_multiple_faces = 0
        self.spoofing_detected = False
        
        # Logging
        self.log_file = os.path.join(self.log_dir, f"proctor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
    def start(self):
        """Start the proctoring system."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        self.log("Proctoring session started")
        self._main_loop()
        
    def stop(self):
        """Stop the proctoring system."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.log("Proctoring session ended")
        
    def _main_loop(self):
        """Main processing loop."""
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                self.frame_count += 1
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow("Online Exam Proctoring", processed_frame)
                
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            self.log(f"Error in main loop: {str(e)}")
        finally:
            self.stop()
            
    def _process_frame(self, frame):
        """Process a single frame.
        
        Args:
            frame: The video frame to process
            
        Returns:
            The processed frame with visualizations
        """
        # Create a copy for visualization
        viz_frame = frame.copy()
        
        # Face detection (not every frame for performance)
        if self.frame_count % FACE_DETECTION_INTERVAL == 0:
            self.faces = find_faces(frame, self.face_model)
            
            # Check number of faces
            self._check_face_count(viz_frame)
            
            # Face spoofing detection
            if len(self.faces) == 1:
                self.spoofing_detected = detect_spoofing(frame, self.faces[0])
                if self.spoofing_detected:
                    self._log_alert("Face spoofing detected")
                    cv2.putText(viz_frame, "SPOOFING DETECTED", (10, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Process each detected face
        for face_rect in self.faces:
            # Draw face rectangle
            x, y, x1, y1 = face_rect
            cv2.rectangle(viz_frame, (x, y), (x1, y1), (0, 255, 0), 2)
            
            # Get facial landmarks
            shape = detect_marks(frame, self.landmark_model, face_rect)
            
            # Eye tracking
            self._track_eyes(frame, viz_frame, shape)
            
            # Mouth opening detection
            self._detect_mouth_opening(shape, viz_frame)
            
            # Head pose estimation
            self._estimate_head_pose(frame, shape, viz_frame)
        
        # Add status information
        self._add_status_info(viz_frame)
        
        return viz_frame
    
    def _check_face_count(self, frame):
        """Check if the correct number of faces is detected."""
        if len(self.faces) == 0:
            self.consecutive_no_face += 1
            if self.consecutive_no_face >= 10:  # About 0.5 seconds at 20 FPS
                self._log_alert("No face detected")
                cv2.putText(frame, "NO FACE DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.consecutive_no_face = 0
            
        if len(self.faces) > 1:
            self.consecutive_multiple_faces += 1
            if self.consecutive_multiple_faces >= 10:
                self._log_alert(f"Multiple faces detected: {len(self.faces)}")
                cv2.putText(frame, "MULTIPLE FACES", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            self.consecutive_multiple_faces = 0
    
    def _track_eyes(self, frame, viz_frame, shape):
        """Track eye movements."""
        # Create eye mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask, left_points = eye_on_mask(mask, self.left_eye_idxs, shape)
        mask, right_points = eye_on_mask(mask, self.right_eye_idxs, shape)
        mask = cv2.dilate(mask, self.kernel, 5)
        
        # Extract eyes region
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        
        # Convert to grayscale and threshold
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(eyes_gray, 75, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        
        # Find eye midpoint
        mid = int((shape[42][0] + shape[39][0]) // 2)
        
        # Detect eyeball positions
        eyeball_pos_left = contouring(thresh[:, 0:mid], mid, viz_frame, left_points)
        eyeball_pos_right = contouring(thresh[:, mid:], mid, viz_frame, right_points, True)
        
        # Check if both eyes are looking in the same direction
        if eyeball_pos_left == eyeball_pos_right and eyeball_pos_left != 0:
            direction = eyeball_pos_left
            
            # Check if direction is the same as last frame
            if direction == self.last_eye_direction:
                self.consecutive_eye_deviations += 1
                if self.consecutive_eye_deviations >= 5:  # Sustained for 5 frames
                    direction_text = {1: "left", 2: "right", 3: "up"}
                    self._log_alert(f"Looking {direction_text[direction]}")
                    cv2.putText(viz_frame, f"Looking {direction_text[direction]}", 
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self.consecutive_eye_deviations = 0
                
            self.last_eye_direction = direction
        else:
            self.consecutive_eye_deviations = 0
            self.last_eye_direction = 0
    
    def _detect_mouth_opening(self, shape, frame):
        """Detect mouth opening."""
        mouth_top = shape[62]
        mouth_bottom = shape[66]
        mouth_height = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
        
        # Initialize reference height if not set
        if self.initial_mouth_height is None and mouth_height > 0:
            self.initial_mouth_height = mouth_height
            return
        
        # Check if mouth is open
        if self.initial_mouth_height and mouth_height > self.initial_mouth_height * 1.5:
            self.mouth_open_count += 1
            if self.mouth_open_count >= 5:  # Sustained for 5 frames
                self._log_alert("Mouth opening detected")
                cv2.putText(frame, "MOUTH OPEN", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            self.mouth_open_count = 0
    
    def _estimate_head_pose(self, frame, shape, viz_frame):
        """Estimate head pose."""
        # Convert shape to numpy array
        shape_np = np.array(shape, dtype=np.float64)
        
        # Get head pose
        image_points = shape_np[[33, 8, 36, 45, 48, 54]]
        pose = get_head_pose(image_points, frame.shape)
        
        # Check for extreme angles
        if abs(pose[0]) > 15 or abs(pose[1]) > 15:
            self._log_alert(f"Head pose deviation - Pitch: {pose[0]:.1f}, Yaw: {pose[1]:.1f}")
            cv2.putText(viz_frame, "HEAD POSE ALERT", (10, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def _add_status_info(self, frame):
        """Add status information to the frame."""
        cv2.putText(frame, f"Alerts: {self.suspicious_count}", (frame.shape[1] - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add timestamp
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, time_str, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _log_alert(self, message):
        """Log an alert message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = f"{timestamp}: {message}"
        self.alerts.append(alert)
        self.suspicious_count += 1
        self.log(f"ALERT: {message}")
        
        # Save a snapshot of the current frame
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                snapshot_path = os.path.join(self.log_dir, 
                                            f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(snapshot_path, frame)
    
    def log(self, message):
        """Log a message to the log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp}: {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Online Exam Proctoring System')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--log-dir', type=str, default=LOG_DIR,
                        help=f'Log directory (default: {LOG_DIR})')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    print("Starting Online Exam Proctoring System...")
    print("Press 'q' to quit")
    
    proctor = OnlineProctor(video_source=args.camera, log_dir=args.log_dir)
    proctor.start()