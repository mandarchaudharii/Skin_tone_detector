import cv2
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained model from Huggingface (you may need to replace this path with the Huggingface model loading code)
@st.cache_resource
def load_model_from_huggingface():
    model = load_model("https://huggingface.co/mandarchaudharii/skin_tone_detection/skin_tone_model.h5")
    return model

# Function to classify skin tone from an image
def classify_skin_tone(image, model):
    resized_img = cv2.resize(image, (224, 224))  # Resize to match model input size
    resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
    resized_img = resized_img / 255.0  # Normalize image

    # Predict skin tone class
    prediction = model.predict(resized_img)
    classes = ["dark", "mid-dark", "mid-light", "light"]
    return classes[np.argmax(prediction)]

# Streamlit App
def main():
    st.title("Live Skin Tone Detector")

    # Load model
    model = load_model_from_huggingface()

    # Access the webcam
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])  # Placeholder for video feed
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access the camera.")
            break

        # Convert the frame to YCrCb color space for skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)

        # Create mask for skin region
        mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        skin = cv2.bitwise_and(frame, frame, mask=mask)

        # Classify the detected skin tone region
        skin_tone = classify_skin_tone(skin, model)

        # Display the appropriate country
        if skin_tone in ["dark", "mid-dark"]:
            country = "Nigeria"
        else:
            country = "India"

        # Display the country on the video feed
        cv2.putText(frame, country, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Update the streamlit image element with the frame
        FRAME_WINDOW.image(frame, channels='BGR')

    camera.release()

if __name__ == '__main__':
    main()
