import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# Load the model from Hugging Face
try:
    model_path = hf_hub_download(repo_id="mandarchaudharii/skin_tone_detection", filename="skin_tone_model.h5")
    model = load_model(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Function to classify skin tone from an image
def classify_skin_tone(image):
    try:
        st.write(f"Original Image Shape: {image.shape}")  # Debugging: Show original image shape

        # Resize image to match model input size
        resized_img = cv2.resize(image, (224, 224))
        st.write(f"Resized Image Shape: {resized_img.shape}")  # Debugging: Show resized image shape
        
        # Normalize the image (ensure this matches how you trained the model)
        resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
        resized_img = resized_img / 255.0  # Normalize image
        
        # Make a prediction
        prediction = model.predict(resized_img)
        
        # Define class labels
        classes = ["dark", "mid-dark", "mid-light", "light"]
        
        # Return the class with the highest probability
        predicted_class = classes[np.argmax(prediction)]
        
        # Debugging: Display prediction probabilities
        st.write(f"Raw prediction probabilities: {prediction}")
        
        return predicted_class, prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

def main():
    st.title("Live Skin Tone Detection")

    # Start video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stframe = st.empty()  # Create an empty placeholder for the video frame

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        # Classify the skin tone
        skin_tone, raw_prediction = classify_skin_tone(frame)

        # Display the frame with prediction
        stframe.image(frame, channels="BGR", caption='Webcam Feed', use_column_width=True)

        # Check if prediction was successful
        if skin_tone is not None:
            st.write(f"Predicted Skin Tone: {skin_tone}")
            st.write(f"Raw Prediction Output: {raw_prediction}")

            # Display the result with the country
            if skin_tone in ["dark", "mid-dark"]:
                st.write("Predicted Country: **Nigeria**")
            else:
                st.write("Predicted Country: **India**")
        
        # Allow for a short delay to make sure the frame is updated properly
        st.time.sleep(0.1)  # Adjust as needed

    # Release the video capture
    cap.release()

if __name__ == "__main__":
    main()
