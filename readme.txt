
Automated Proctoring System: This project endeavors to create an automated proctoring 
system that monitors users via webcam and microphone during online examinations.
It comprises two main components: vision-based and audio-based functionalities.

Below, we detail the functionalities of the vision-based component without relying 
on external references:

Prerequisites: Before running the programs in this repository, set up a virtual 
environment following these steps:
1.Create a virtual environment: python -m venv venv
2.Activate the virtual environment: cd ./venv/Scripts/activate (Windows users)
source ./venv/bin/activate (Mac and Linux users)
3.Install the requirements: pip install --upgrade pip (to upgrade pip)
pip install -r requirements.txt

Features:

1.Eye tracking: This feature detects left, right, or upward eye movements, 
enabling monitoring of users' gaze directions.

2.Mouth opening detection: By recording initial lip distances, the system 
identifies instances of mouth opening during examinations, ensuring compliance 
with exam rules.

3.Instance segmentation: This function counts individuals and distinguishes if no 
one or more than one person is detected in the exam environment, maintaining exam integrity.

4.Head pose estimation: By analyzing head orientation, this feature determines the user's 
gaze direction, aiding in monitoring their focus during the exam.

5.Face spoofing detection: Using advanced algorithms, the system identifies and flags any 
attempts at face spoofing or presenting false images during the exam.

Face Detection: The project compares different face detection models, with OpenCV's DNN 
module providing the best results. The implementation is in face_detector.py, facilitating 
various functionalities within the system.

Facial Landmarks: A Tensorflow-based model is utilized for facial landmark detection, 
replacing Dlib's model for improved accuracy. The implementation can be found in face_landmarks.py, 
enabling precise tracking of facial features.

