import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load the saved model
model = load_model("digit_classifier_model.keras")

# Title and description
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) or draw a digit on the canvas, and the model will predict the digit.")

# Option to upload an image
option = st.selectbox("Choose input method", ["Upload Image", "Draw on Canvas"])

if option == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_resized = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image_resized) / 255.0  # Normalize pixel values
        image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for the model

        # Predict
        prediction = model.predict(image_array).argmax()
        st.write(f"Predicted Digit: **{prediction}**")

elif option == "Draw on Canvas":
    # Create a drawing canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Black color for drawing
        stroke_width=10,                # Thickness of the stroke
        stroke_color="white",           # White for the digit
        background_color="black",       # Black background
        height=280,                     # Canvas height
        width=280,                      # Canvas width
        drawing_mode="freedraw",        # Free drawing mode
        key="canvas",
    )

    if canvas_result.image_data is not None and canvas_result.image_data.any():
    # Process the drawing
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")  # Convert RGBA to grayscale
        img_resized = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img_resized) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model

        # Display the processed image
        st.write("Processed Image:")
        st.image(img_resized, width=150)

        # Make a prediction
        prediction = model.predict(img_array).argmax()
        probabilities = model.predict(img_array).flatten()

        # Display the prediction
        st.write(f"Predicted Digit: **{prediction}**")
    else:
        st.write("Please draw a digit on the canvas.")
