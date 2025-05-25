import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib


MODEL_PATH = 'saved_models/rf_robust_digit_classifier_v1.joblib' 
PREPROCESSOR_FUNCTION_NAME = 'preprocess_for_lighting'

from image_preprocessor import preprocess_for_lighting as digit_preprocessor

# Page Configuration 
st.set_page_config(
    page_title="Digit Classifier",
    page_icon="",
    layout="wide"
)

# Load Trained Model
@st.cache_resource 
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"ERROR: Model file not found at '{path}'. Classification will not be available.")
        return None
    except Exception as e:
        st.error(f"ERROR: Could not load model: {e}. Classification will not be available.")
        return None

rf_model = load_model(MODEL_PATH)

# --- Main Application UI ---
st.title("✏️ Handwritten Digit Classifier")
st.markdown("Upload an image of a single handwritten digit (0-9). The app will attempt to classify it, even with some lighting variations.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your image here:",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)

        st.subheader("Image Analysis & Classification")
        col1, col2, col3 = st.columns([2,2,3]) 

        with col1:
            st.markdown("###### Uploaded Image")
            st.image(pil_image, use_container_width=True)

        # Check if model and preprocessor are ready
        if rf_model is not None and callable(digit_preprocessor):
            with st.spinner("Processing and classifying..."):
                # Convert PIL Image to Grayscale NumPy array
                image_np_array = np.array(pil_image.convert('L'))

                # Resize to 28x28 and ensure uint8 (as expected by preprocessor)
                resized_image_np_uint8 = cv2.resize(image_np_array, (28, 28), interpolation=cv2.INTER_AREA).astype(np.uint8)

                # Preprocess the image
                processed_features = digit_preprocessor(resized_image_np_uint8)

                if processed_features is not None:
                    with col2:
                        st.markdown("###### Processed for Model")
                        processed_image_display = processed_features.reshape(28, 28)
                        st.image(processed_image_display, use_container_width=True, clamp=True)
                    
                    # Prediction Step 
                    # Reshape features for a single sample prediction (model expects 2D array)
                    single_sample_features = processed_features.reshape(1, -1)
                    
                    # Get probability scores for each class (digit 0-9)
                    prediction_probabilities = rf_model.predict_proba(single_sample_features)
                    
                    # Get the digit with the highest probability
                    predicted_digit = np.argmax(prediction_probabilities)
                    
                    # Get the confidence score for the predicted digit
                    confidence = prediction_probabilities[0, predicted_digit]

                    with col3:
                        st.markdown("###### Classification Result")
                        st.metric(label="Predicted Digit", value=str(predicted_digit))
                        
                        # Display confidence as a progress bar and text
                        st.write("Confidence:")
                        st.progress(float(confidence)) 
                        st.caption(f"{confidence*100:.1f}%")

                        # Display top N predictions with probabilities
                        st.markdown("---")
                        st.write("Top Probabilities:")
                        # Get top 3 predictions (example)
                        top_n_indices = np.argsort(prediction_probabilities[0])[::-1][:3]
                        for i in top_n_indices:
                            st.write(f"Digit {i}: {prediction_probabilities[0, i]*100:.1f}%")

                else: # Preprocessing failed
                    with col2: 
                        st.error("Could not preprocess the image. Please try a different one.")
        else: 
            if rf_model is None : st.warning("Model not available for classification.")
            if not callable(digit_preprocessor) : st.warning("Preprocessor not available for classification.")


    except PIL.UnidentifiedImageError:
        st.error(f"Cannot identify image file: '{uploaded_file.name}'. Please upload a valid PNG, JPG, or JPEG image.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

else: # No file uploaded
    st.info("Awaiting image upload.")
    st.markdown("""
    **Instructions:**
    1. Click on "Browse files" or drag and drop an image.
    2. Ensure the image contains a single, clearly visible handwritten digit.
    """)

#  Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Random Forest model and image preprocessing "
    "to classify handwritten digits, aiming for robustness against varied lighting."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Technical Details")
st.sidebar.markdown(f"""
- **Model:** Random Forest
- **Training Data:** MNIST (augmented)
- **Key Preprocessing:** Adaptive Thresholding
""")