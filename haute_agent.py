import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from llama_index.llms.groq import Groq
from llama_index.core.settings import Settings
import mediapipe as mp
from dotenv import load_dotenv
from facenet_pytorch import MTCNN

# Page config
st.set_page_config(
    page_title="Haute-U AR",
    page_icon="👗",
    layout="wide"
)

# Minimalist CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main {
        background-color: #ffffff;
        padding: 1rem;
    }
    .stButton>button {
        background-color: #000000;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #333333;
    }
    .uploadedFile {
        border: 1px solid #000000;
        border-radius: 4px;
        padding: 8px;
    }
    h1, h2, h3 {
        color: #000000;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .chat-message {
        padding: 0.8rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f5f5f5;
    }
    .ai-message {
        background-color: #f0f0f0;
    }
    .reset-button {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 2rem;
    }
    .header-container img {
        margin-bottom: 0 !important;
    }
    .header-container h1 {
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Reset function
def reset_chat():
    st.session_state.image_uploaded = False
    st.session_state.chat_history = []
    st.session_state.skin_analysis = None
    st.session_state.marked_image = None
    st.rerun()

# Add reset button
if st.session_state.image_uploaded:
    if st.button("New Analysis", key="reset", help="Start a new analysis"):
        reset_chat()

# Load environment variables and initialize APIs
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set your Groq API key in the environment variables (GROQ_API_KEY)")
    st.stop()

llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
Settings.llm = llm

# Initialize MediaPipe Face Mesh and MTCNN
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mtcnn = MTCNN(keep_all=True, device='cpu')


def detect_and_mark_dark_spots(image):
    """Detect dark spots and mark them on the image."""
    img_rgb = np.array(image)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Apply CLAHE for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(enhanced_gray, (11, 11), 0)
    
    # Dynamic thresholding
    mean_intensity = np.mean(enhanced_gray)
    thresh_value = mean_intensity * 0.7
    _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_spot_area = 20  # Minimum area to be considered a spot
    max_spot_area = 500  # Max area to avoid false positives
    dark_spots = [c for c in contours if min_spot_area < cv2.contourArea(c) < max_spot_area]

    # Draw dark spot markers on image
    image_with_spots = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(image_with_spots)
    for c in dark_spots:
        x, y, w, h = cv2.boundingRect(c)
        draw.rectangle([x, y, x+w, y+h], outline="red", width=2)

    return image_with_spots, len(dark_spots)


def get_skin_analysis(image):
    """Analyze skin and generate a detailed report."""
    img_array = np.array(image)
    
    # Face detection
    boxes, _ = mtcnn.detect(image)
    if boxes is None:
        return None, None

    # Crop face
    box = boxes[0]
    x, y, w, h = [int(coord) for coord in box]
    face_region = img_array[y:h, x:w]

    if face_region.shape[0] == 0 or face_region.shape[1] == 0:
        return None, None

    # Convert to HSV and YCrCb
    face_hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
    face_ycrcb = cv2.cvtColor(face_region, cv2.COLOR_RGB2YCrCb)

    # Get average color values
    hsv_means = cv2.mean(face_hsv)
    ycrcb_means = cv2.mean(face_ycrcb)

    # Detect and mark dark spots
    marked_image, dark_spots_count = detect_and_mark_dark_spots(image)

    # Generate analysis
    analysis_prompt = f"""
    As a dermatologist, analyze this skin tone data:
    - Brightness Level: {ycrcb_means[0]:.2f}
    - Redness Level: {ycrcb_means[1]:.2f}
    - Dark Spots Count: {dark_spots_count}

    Provide an expert skin tone assessment and suitable fashion recommendations.
    """
    response = llm.complete(analysis_prompt)
    return response.text, marked_image


def get_fashion_recommendations(skin_analysis, user_query=None):
    """Generate fashion advice based on skin analysis and user input."""
    base_prompt = f"""
    As a luxury fashion consultant, provide concise recommendations based on this skin tone analysis:
    
    {skin_analysis}

    Focus on:
    - Best colors
    - Key styles
    - Quick makeup tips
    
    Keep the response under 1500 characters.
    Question: {user_query if user_query else "General advice"}
    """
    response = llm.complete(base_prompt)
    return response.text[:1500]  # Ensure response is truncated


# Main UI
st.image("logo.png", width=25)
st.markdown("""
    <div class="header-container">
        <img>
        <h1>Hue Style AI</h1>
       
            Powered by Haute-U AR technologies
    </div>
""", unsafe_allow_html=True)

if not st.session_state.image_uploaded:
    uploaded_file = st.file_uploader("Upload a clear photo of your face", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode in ["RGBA", "P"]:
            image = image.convert("RGB")
        st.session_state.image_uploaded = True
        st.session_state.current_image = image
        st.rerun()

if st.session_state.image_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.image(st.session_state.current_image, caption="Original Image", use_container_width=True)

    if st.session_state.get("skin_analysis") is None:
        with st.spinner("Analyzing..."):
            try:
                analysis, marked_img = get_skin_analysis(st.session_state.current_image)
                if analysis:
                    st.session_state.skin_analysis = analysis
                    st.session_state.marked_image = marked_img
                else:
                    st.error("No face detected. Please try again.")
                    st.session_state.image_uploaded = False
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.image_uploaded = False

    with col2:
        if st.session_state.get("marked_image"):
            st.image(st.session_state.marked_image, caption="Analysis", use_container_width=True)

    if st.session_state.get("skin_analysis"):
        st.markdown("""
            <div style="background-color: white; padding: 15px; margin-top: 20px;">
                <h3>Analysis Results</h3>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 15px; margin-bottom: 20px;">
                {st.session_state.skin_analysis}
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<h3>Fashion Consultation</h3>", unsafe_allow_html=True)

        for message in st.session_state.chat_history:
            role = "You" if message["role"] == "user" else "AI Stylist"
            css_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f"""
                <div class="chat-message {css_class}">
                    <strong>{role}:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)

        user_input = st.text_input("Ask your stylist:", key="user_input")
        if st.button("Get Advice"):
            if user_input:
                with st.spinner("Creating recommendations..."):
                    response = get_fashion_recommendations(st.session_state.skin_analysis, user_input)
                    st.session_state.chat_history.append(
                        {"role": "user", "content": user_input}
                    )
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                st.rerun()

# Simplified Tips
with st.expander("Tips"):
    st.markdown("""
    - Use natural lighting
    - Center your face
    - Clean, makeup-free skin
    - Keep ~2 feet distance
    """)