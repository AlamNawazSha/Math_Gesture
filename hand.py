# You have to downlaed old version of python which is 3.10.0
# and import all the packegs like mediapipe cvzone cv2 ect..
# https://youtu.be/RhhN0CLnFdc?si=6DlAiyOfdEWQgwye this youtude link
# issue in webcame width
import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
from google.generativeai.types import content_types
import streamlit as st 

st.set_page_config(layout="wide")
st.image('MathGestures.png')

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run',value=True)
    FRAME_WINDOW = st.image([])

with col2:
    outpit_text_area = st.title("Answer")
    outpit_text_area = st.subheader("")

genai.configure(api_key="AIzaSyAqAhMd62MvuXv88EBOxzBrDy2PuFgGEJM")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand)
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info,prev_pos,canvas):
    fingers,lmList = info
    current_pos = None
    if fingers[1] == 1 and fingers.count(1) == 1:
         current_pos = lmList[8][0:2]
         if prev_pos is None : prev_pos = current_pos
         cv2.line(canvas,current_pos,prev_pos,(255,0,255),10)
    elif fingers  ==  [1,1,1,1,1]:
        canvas = np.zeros_like(img)     
        
        
    return current_pos , canvas 
       
    
    # Continuously get frames from the webcam
    
# def sendToAI(model,canvas,fingers):
#     if fingers == [1,1,1,1,0]:
#         pil_image = Image.fromarray(canvas)
#         response = model.generate_content([
#             {"text": "Solve the Math Problem"},
#             {"image": pil_image}])

#         # response = model.generate_content("Whic is the largest ocean?")
#         return response.text

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        # Check if the canvas is NOT blank (i.e., user has drawn something)
        if not np.any(canvas):
            print("Canvas is blank, skipping AI request.")
            return "🖼️ Please draw something before asking AI to solve it."

        try:
            # Convert to RGB (OpenCV is BGR)
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(canvas_rgb)

            response = model.generate_content([
                {"text": "Solve the Math Problem Drawon in this image."},
                {"image": pil_image}
            ])
            return response.text
        except Exception as e:
            print(f"AI Error: {e}")
            return "⚠️ AI could not process the image. Try again."





prev_pos = None     
canvas = None
image_combines = None
output_tex = ""

while True:
# Capture each frame from the webcam
# 'success' will be True if the frame is successfully captured, 'img' will contain the frame
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
       
    
    
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_tex = sendToAI(model,canvas,fingers)

        
    image_combines = cv2.addWeighted(img,0.5,canvas,0.3,0)
    
    # Display the image in a window
    FRAME_WINDOW.image(image_combines,channels="BGR")
    if output_tex:
        outpit_text_area.text(output_tex)
    # cv2.imshow("Image", img)
    # cv2.imshow("Canvas", canvas)
    # cv2.imshow("Image_combines", image_combines)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    cv2.waitKey(1)
