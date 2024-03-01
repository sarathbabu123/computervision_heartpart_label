# Importing the necessary libraries
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import tkinter as tk
from PIL import Image, ImageTk

# Load the heart image
image = cv2.imread("./heart-diagram - Copy.png")
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Function to define the lower and upper bounds of a color in bgr format
def limit(color):
    c = np.uint8([[color]])
    hsv_color = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_color[0][0][0] - 10, 100, 100])
    upper = np.array([hsv_color[0][0][0] + 10, 255, 255])
    return lower, upper

# Defining the lower and upper bounds for different colors
red_lower, red_upper = limit([0, 0, 192])
ltred_lower,ltred_upper = limit([69, 62, 211])
blue_lower,blue_upper = limit([227, 179, 131])

# Define BGR values for masks and masking the parts septum and right atriium based on the bgr value of the colour
septum_bgr_value = [171, 171, 248]
septummask = np.all(image == septum_bgr_value, axis=2).astype(np.uint8) * 255
rtatrium_bgr_value = [234, 202, 174]
rtatriummask = np.all(image == rtatrium_bgr_value, axis=2).astype(np.uint8) * 255

# Initialize Mediapipe drawing utilities and hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Opening webcam and getting the frame widtha and height
cap = cv2.VideoCapture(0)
fr_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fr_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize text-to-speech engine

engine = pyttsx3.init()

# Function to speak text

def speak(text):
    if not engine._inLoop:
        engine.say(text)
        engine.runAndWait()

# Function to handle body parts
def handle_part(part_name):
    threading.Thread(target=speak, args=(part_name,)).start()

# Initialize Mediapipe Hands module
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    # Create Tkinter GUI window
    root = tk.Tk()
    root.title("Webcam and Image")
    
    # Create labels to display the images
    image_label = tk.Label(root)
    image_label.pack(padx=10, pady=10)
    
    webcam_label = tk.Label(root)
    webcam_label.pack(padx=10, pady=10)
    
    # Main loop to capture frames from webcam and ensuring realtime tracking on the image
    while cap.isOpened():
        # Read a frame from the webcam
        _, frame = cap.read()
        if frame is None:
            print("No camera detected !!")
            break
        # Create a copy of the original image everytime while loop run to erasr the tracking marks on previous frame  
        image_copy = image.copy()
        height, width, _ = image_copy.shape
        print(image_copy.shape)
        print(frame.shape)
        # Convert the webcam frame from BGR to RGB color space
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # Process the frame using Mediapipe Hands module
        result = hands.process(frame)
        
        # Apply color masks to the image. To separate the left and right parts
        red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
        lightred_mask = cv2.inRange(hsv_image, ltred_lower, ltred_upper)
        blue_mask = cv2.inRange(hsv_image,blue_lower,blue_upper)
        
        # Find the contours of the masks
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ltredcontours,_ = cv2.findContours(lightred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        septcontours, _ = cv2.findContours(septummask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bluecontours,_ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rtatriumcontours,_ = cv2.findContours(rtatriummask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if hands are detected in the webcam frame
        if result.multi_hand_landmarks is not None:
            # Get the normalized coordinates of the index finger tip
            normalized_landmark = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            # denormalize it for the  image size
            pixel_coordinates_landmark = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, width, height)
            # # denormalizing the coorsinates for the frame
            # cam_denorm_coordinates = mp_drawing._normalized_to_pixel_coordinates(normalized_landmark.x, normalized_landmark.y, fr_width, fr_height)
            
            # Draw a circle at the detected finger tip position on the webcam frame
            cv2.circle(frame,pixel_coordinates_landmark,3,(0,255,0),3)
            # Iterate through the contours and perform actions based on their areas
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area == 5550:
                    # Draw contour and circle on the image copy
                    # cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        # Speak the body part name associated with the contour
                        handle_part("aorta")
                        print("aorta")
                elif area == 5045:
                    # cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Left Atrium")
                        print("Left Atrium")
                elif area == 1173:
                    # cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Descending Aorta")
                        print("Descending Aorta")
                elif area == 786:
                    # cv2.drawContours(image_copy, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Pulmonary Vein")
                        print("Pulmonary Vein")
                        
            for cnt in ltredcontours:
                area = cv2.contourArea(cnt)
                if area == 6684.0:
                    cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Left Ventricle")
                        print("Left Ventricle")
                        
            for cnt in septcontours:
                # cv2.drawContours(image_copy, cnt, -1, (0, 255, 0), 2)
                if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                    handle_part("Septum")
                    print("Septum")
                    
            for cnt in bluecontours:
                area = cv2.contourArea(cnt)
                cv2.circle(image_copy, pixel_coordinates_landmark, 10, (0, 255, 0), 2)
                if area == 1334.0:
                    # cv2.drawContours(image_copy,cnt,-1, (255,0,0),2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Inferior Venacava")
                        print("Inferior Venacava")
                elif (area == 3.5 or area == 117.5):
                    # cv2.drawContours(image_copy,cnt,-1, (255,0,0),2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Pulmonary valve")
                        print("Pulmonary valve")
                elif area == 6879.5:
                    # cv2.drawContours(image_copy,cnt,-1, (255,0,0),2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Right Ventricle")
                        print("Right Ventricle")
                elif area == 4015.5:
                    # cv2.drawContours(image_copy,cnt,-1, (255,0,0),2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Pulmonary artery")
                        print("Pulmonary artery")
                elif area == 4256.0:
                    # cv2.drawContours(image_copy,cnt,-1, (255,0,0),2)
                    if cv2.pointPolygonTest(cnt, pixel_coordinates_landmark, False) >= 0:
                        handle_part("Venacava")
                        print("Venacava")
            for cnt in rtatriumcontours:
                # cv2.drawContours(image_copy,cnt,-1,(255,0,255),2)
                if  cv2.pointPolygonTest(cnt,pixel_coordinates_landmark, False) >= 0:
                    handle_part("Right atrium")
                    print("Right atrim")
        
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize and display webcam frame on the GUI
        webcam_img = Image.fromarray(cv2.resize(frame, (int(fr_width/2), int(fr_height/2))))
        webcam_img = ImageTk.PhotoImage(image=webcam_img)
        webcam_label.config(image=webcam_img)
        webcam_label.image = webcam_img
        
        # Resize and display image frame on the GUI
        img_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        img_copy = cv2.resize(img_copy, (int(width), int(height)))  
        image_img = Image.fromarray(img_copy)
        image_img = ImageTk.PhotoImage(image=image_img)
        image_label.config(image=image_img)
        image_label.image = image_img
        
        # Update Tkinter GUI
        root.update_idletasks()
        root.update()
        
        # Check for key press 'q' to exit the loop
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    
# Release the webcam and close OpenCV windows    
cap.release()
cv2.destroyAllWindows()
# Close the Tkinter GUI window
root.mainloop()