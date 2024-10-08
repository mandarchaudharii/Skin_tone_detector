import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# Load the model from Hugging Face
model_path = hf_hub_download(repo_id="mandarchaudharii/skin_tone_detection", filename="skin_tone_model.h5")

# Load the .h5 model using TensorFlow/Keras
model = load_model(model_path)

# Function to classify skin tone from an image
def classify_skin_tone(image):
    resized_img = cv2.resize(image, (224, 224))
    resized_img = np.expand_dims(resized_img, axis=0)
    resized_img = resized_img / 255.0  # Normalize the image
    prediction = model.predict(resized_img)
    classes = ["dark", "mid-dark", "mid-light", "light"]
    return classes[np.argmax(prediction)]

def main():
    st.title("Skin Tone Detection")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Show the uploaded image
        st.image(img, channels="BGR", caption='Uploaded Image', use_column_width=True)

        # Classify the skin tone
        skin_tone = classify_skin_tone(img)

        st.write(skin_tone)

        # Display the result with the country
        if skin_tone in ["dark", "mid-dark"]:
            st.write("Predicted Country: **Nigga**")
        else:
            st.write("Predicted Country: **White**")

if __name__ == "__main__":
    main()
