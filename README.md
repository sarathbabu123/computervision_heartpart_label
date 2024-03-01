# Real-time Heart Part Detection on Heart Diagram
Real-time heart part detection &amp; tracking using Mediapipe &amp; OpenCV. Overlay live webcam feed with heart diagram. Hand movement highlights corresponding part on diagram &amp; speaks its name.

## Installation
Clone this repository to your local machine.
Install the required dependencies by running pip install -r requirements.txt.
Make sure you have a webcam connected to your system.
## Usage
Run the script heartpart_cv.py.
Place your hand in front of the webcam.
Move your hand over different parts of the heart diagram.
As each part is detected, its name will be spoken aloud.
Press the 'q' key to exit the application.
## Requirements
Python 3.x
OpenCV
Mediapipe
NumPy
pyttsx3
tkinter
Pillow
## Design Decisions
The project uses the Mediapipe Hands module for hand tracking, allowing precise detection of hand landmarks.
Color masking is applied to isolate different parts of the heart diagram based on their colors.
Text-to-speech functionality is implemented using the pyttsx3 library to provide audio feedback on detected body parts.
## Bugs Identified
No known bugs at the moment.
## Future Optimization Suggestions
Implementing multi-hand tracking to support detection and tracking of both hands simultaneously.
Enhancing the accuracy of body part detection through model fine-tuning or additional preprocessing techniques.
Optimizing code efficiency for real-time performance, especially in the contour detection and processing steps.
## Credits
This project was developed by Sarath babu P as part of internship related assignment at dVerse Technologies. Special thanks to the creators of the Mediapipe library and OpenCV for their invaluable contributions.
