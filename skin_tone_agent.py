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

# Custom styling
st.set_page_config(
    page_title="Haute-U AR Technologies",
    page_icon="👗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff5f7 0%, #fff0f7 100%);
    }
    .main {
        background-color: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 20px;
    }
    .stButton>button {
        background-color: #FF1493;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF69B4;
    }
    .uploadedFile {
        border: 2px dashed #FF1493;
        border-radius: 10px;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #FF1493;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #FFE4E1;
    }
    .ai-message {
        background-color: #FFF0F5;
    }
    .nav-button {
        width: 100%;
        padding: 10px;
        margin: 5px 0;
        background-color: #FF1493;
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        display: block;
    }
    .nav-button:hover {
        background-color: #FF69B4;
    }
    .sidebar {
        position: fixed;
        left: 0;
        top: 0;
        padding: 2rem 1rem;
        width: 200px;
        height: 100vh;
        background-color: white;
        box-shadow: 2px 0 5px rgba(0,0,0,0.1);
    }
    .content {
        margin-left: 220px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for navigation
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Skin Analysis'

# Create sidebar navigation
st.markdown("""
    <div class="sidebar">
        <h3 style="text-align: center; margin-bottom: 20px;">Navigation</h3>
        <a href="?tab=skin" class="nav-button" id="skin-tab">🔍 Skin Analysis</a>
        <a href="?tab=fashion" class="nav-button" id="fashion-tab">👗 Fashion Consultation</a>
    </div>
""", unsafe_allow_html=True)

# Main content area
st.markdown('<div class="content">', unsafe_allow_html=True)

# Load environment variables and initialize APIs (same as before)
load_dotenv()

# Initialize Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please set your Groq API key in the environment variables (GROQ_API_KEY)")
    st.stop()

llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
Settings.llm = llm

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Initialize MTCNN for face detection
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
    As a dermatologist, analyze this skin data:
    - Brightness Level: {ycrcb_means[0]:.2f}
    - Redness Level: {ycrcb_means[1]:.2f}
    - Dark Spots Count: {dark_spots_count}

    Provide an expert skin condition assessment and suitable skincare recommendations.
    """
    response = llm.complete(analysis_prompt)
    return response.text, marked_image


def get_fashion_recommendations(skin_analysis, user_query=None):
    """Generate fashion advice based on skin analysis and user input."""
    base_prompt = f"""
    You are a luxury fashion consultant. Provide color and style recommendations based on this skin analysis:
    
    {skin_analysis}

    Suggestions should include:
    - Most flattering colors
    - Clothing styles and accessories
    - Makeup recommendations
    - Seasonal trends

    If the client has a specific question, focus on that:
    {user_query if user_query else "General advice"}
    """
    response = llm.complete(base_prompt)
    return response.text


if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Skin Analysis'

# Create sidebar navigation
st.markdown("""
    <div class="sidebar">
        <h3 style="text-align: center; margin-bottom: 20px;">Navigation</h3>
        <a href="?tab=skin" class="nav-button" id="skin-tab">🔍 Skin Analysis</a>
        <a href="?tab=fashion" class="nav-button" id="fashion-tab">👗 Fashion Consultation</a>
    </div>
""", unsafe_allow_html=True)

# Main content area
st.markdown('<div class="content">', unsafe_allow_html=True)

# Logo and title
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.image("logo_1.svg", width=300)
    st.title("Haute-U AR Technologies")
    st.subheader("AI-Powered Fashion & Skin Analysis")

# Get current tab from URL parameters
current_tab = st.query_params.get("tab", "skin")

if current_tab == "skin":
    uploaded_file = st.file_uploader("Upload a clear photo of your face", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        
        if image.mode in ["RGBA", "P"]:
            image = image.convert("RGB")
            
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        
        if st.session_state.get("skin_analysis") is None:
            with st.spinner("✨ Analyzing your skin characteristics..."):
                try:
                    analysis, marked_img = get_skin_analysis(image)
                    if analysis:
                        st.session_state.skin_analysis = analysis
                        st.session_state.marked_image = marked_img
                    else:
                        st.error("No face detected. Please upload a clearer image.")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
        
        with col2:
            if st.session_state.get("marked_image"):
                st.image(st.session_state.marked_image, caption="Analysis Results", use_container_width=True)
        
        if st.session_state.get("skin_analysis"):
            st.markdown("""
                <div style="background-color: white; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <h3 style="color: #FF1493; margin-bottom: 15px;">Your Skin Analysis Results 🔍</h3>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style="background-color: #FFF0F5; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    {st.session_state.skin_analysis}
                </div>
            """, unsafe_allow_html=True)
            
            # Direct link to Fashion Consultation
            st.markdown("""
                <div style="text-align: center; margin-top: 20px;">
                    <a href="?tab=fashion" class="nav-button" style="display: inline-block; width: auto; padding: 15px 30px;">
                        Continue to Fashion Consultation 👗
                    </a>
                </div>
            """, unsafe_allow_html=True)

elif current_tab == "fashion":
    if st.session_state.get("skin_analysis"):
        st.write("### Your Personalized Fashion Consultation")
        
        for message in st.session_state.get("chat_history", []):
            role = "You" if message["role"] == "user" else "AI Fashion Stylist"
            css_class = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f"""
                <div class="chat-message {css_class}">
                    <strong>{role}:</strong><br>
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
        
        user_input = st.text_input("Ask your fashion stylist:", key="user_input")
        if st.button("Get Recommendations 👗"):
            if user_input:
                with st.spinner("Creating your personalized recommendations..."):
                    response = get_fashion_recommendations(st.session_state.skin_analysis, user_input)
                    st.session_state.chat_history = st.session_state.get("chat_history", []) + [
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ]
                st.rerun()
    else:
        st.info("Please complete the skin analysis first to receive personalized fashion recommendations.")
        st.markdown("""
            <div style="text-align: center; margin-top: 20px;">
                <a href="?tab=skin" class="nav-button" style="display: inline-block; width: auto; padding: 15px 30px;">
                    Go to Skin Analysis 🔍
                </a>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Enhanced Tips Section remains at the bottom
with st.expander("✨ Pro Tips for Best Results"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Photo Tips
        - 📸 Use natural lighting
        - 🎯 Center your face in the frame
        - 🧴 Clean, makeup-free skin
        - 📏 Keep about 2 feet distance
        """)
    with col2:
        st.markdown("""
        ### Consultation Tips
        - 👗 Specify your style preferences
        - 🎨 Mention color preferences
        - 📅 Include occasion details
        - 💭 Share specific concerns
        """)