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
    st.title("Skin Tone Detection")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert uploaded file to OpenCV format
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Show the uploaded image
            st.image(img, channels="BGR", caption='Uploaded Image', use_column_width=True)

            # Classify the skin tone
            skin_tone, raw_prediction = classify_skin_tone(img)

            # Check if prediction was successful
            if skin_tone is not None:
                st.write(f"Predicted Skin Tone: {skin_tone}")
                st.write(f"Raw Prediction Output: {raw_prediction}")

                # Display the result with the country
                if skin_tone in ["dark", "mid-dark"]:
                    st.write("Predicted Country: **Nigeria**")
                else:
                    st.write("Predicted Country: **India**")
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
