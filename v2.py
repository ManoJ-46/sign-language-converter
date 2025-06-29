import streamlit as st
import cv2
import mediapipe as mp
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import base64
from CNNModel import CNNModel

#this should be at the top only, do not change here pa
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

#mediapipe setup here
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

#models with cachine
@st.cache_resource
def load_models():
    model_alpha = CNNModel()
    model_alpha.load_state_dict(torch.load("trained.pth", map_location="cpu"))
    model_alpha.eval()
    return model_alpha

model_alpha = load_models()

#mapping of the classes to the alphabet
alphabet_classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# to add background image this space is given





#custom css , you can change font and size here pa
st.markdown(
    """
    <style>
    /* Custom font */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    
    * {
        font-family: 'Montserrat', sans-serif !important;
    }
    
    /* Remove cursor blinking in dropdown */
    .stSelectbox:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    /* Button animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .stButton>button {
        animation: pulse 2s infinite;
        transition: all 0.3s ease;
        border-radius: 12px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Webcam container */
    .webcam-container {
        max-width: 640px;
        margin: 0 auto;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        position: relative;
    }
    
    .fullscreen-btn {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 100;
        background: rgba(0,0,0,0.5);
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }
    
    /* Card styling */
    .card {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: none;
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Prediction badge */
    .prediction-badge {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 30px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

#predicting function
def predict_sign_realtime(frame, model, classes_reverse):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    
    coordinates = []
    x_coords = []
    y_coords = []
    z_coords = []
    predicted_character = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            data = {}
            
            for i in range(len(hand_landmarks.landmark)):
                lm = hand_landmarks.landmark[i]
                x_coords.append(lm.x)
                y_coords.append(lm.y)
                z_coords.append(lm.z)
            
            for i, landmark in enumerate(mp_hands.HandLandmark):
                lm = hand_landmarks.landmark[i]
                data[f'{landmark.name}_x'] = lm.x - min(x_coords)
                data[f'{landmark.name}_y'] = lm.y - min(y_coords)
                data[f'{landmark.name}_z'] = lm.z - min(z_coords)
            
            coordinates.append(data)
            
            h, w, _ = frame.shape
            x1 = int(min(x_coords) * w) - 10
            y1 = int(min(y_coords) * h) - 10
            x2 = int(max(x_coords) * w) + 10
            y2 = int(max(y_coords) * h) + 10
            
            coordinates_df = pd.DataFrame(coordinates)
            coords_reshaped = np.reshape(coordinates_df.values, (coordinates_df.shape[0], 63, 1))
            coords_tensor = torch.from_numpy(coords_reshaped).float()
            
            with torch.no_grad():
                outputs = model(coords_tensor)
                _, predicted = torch.max(outputs.data, 1)
                pred_idx = predicted.cpu().numpy()[0]
                predicted_character = classes_reverse[pred_idx]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
            return frame, predicted_character
    
    return frame, None

#to load sign images
def load_sign_image(letter):
    """Load sign language image from alphabets folder"""
    image_path = f"alphabets/{letter}.jpg"
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        # Fallback to placeholder if image not found
        img = Image.new('RGB', (200, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("arial.ttf", 100)
        except:
            font = ImageFont.load_default()
        
        text_width = draw.textlength(letter, font=font)
        x = (200 - text_width) // 2
        y = 50
        
        draw.text((x, y), letter, fill='black', font=font)
        return img

# Streamlit App UI
st.markdown('<p class="main-title">Sign Language Recognition System</p>', unsafe_allow_html=True)

# Mode selection dropdown
app_mode = st.sidebar.selectbox(
    "Select Mode:",
    ["Live Detection", "English to Sign Language"],
    key="mode_selector"
)

#main content of the page is here
if app_mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Sign Language Detection")
    
    # Initialize session state
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Camera controls with validation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Camera", use_container_width=True):
            st.session_state.camera_active = True
            st.success("Camera started successfully!")
    
    with col2:
        if st.button("Stop Camera", use_container_width=True):
            st.session_state.camera_active = False
            st.info("Camera stopped")
    
    # Camera feed container
    webcam_container = st.container()
    
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera. Check your device connection.")
            st.session_state.camera_active = False
        else:
            frame_placeholder = webcam_container.empty()
            prediction_placeholder = st.empty()
            
            with webcam_container:
                st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
                
                # Fullscreen button
                st.markdown(
                    '<button class="fullscreen-btn" onclick="this.parentElement.requestFullscreen()">⛶</button>',
                    unsafe_allow_html=True
                )
                
                while st.session_state.camera_active and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from camera.")
                        break
                    
                    frame = cv2.flip(frame, 1)
                    processed_frame, prediction = predict_sign_realtime(frame, model_alpha, alphabet_classes)
                    
                    # Display processed frame
                    frame_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    # Display prediction
                    if prediction:
                        prediction_placeholder.markdown(
                            f'<div class="prediction-badge" style="background-color: #4CAF50; color: white;">Prediction: {prediction}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        prediction_placeholder.info("No hand detected")
                
                st.markdown('</div>', unsafe_allow_html=True)
                cap.release()
    else:
        webcam_container.info("Click 'Start Camera' to begin live detection")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:  # English to Sign Language
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("English to Sign Language Converter")
    
    # Text input
    user_text = st.text_input(
        "Enter text to convert to sign language:",
        placeholder="Type your message here...",
        help="Only alphabetic characters will be converted"
    )
    
    # Convert button
    if st.button("Convert to Sign Language", use_container_width=True):
        if not user_text:
            st.error("Please enter some text to convert!")
        else:
            # Filter only alphabetic characters
            filtered_text = ''.join([char.upper() for char in user_text if char.isalpha()])
            
            if filtered_text:
                st.subheader(f"Sign Language for: '{user_text}'")
                st.caption(f"Filtered letters: {filtered_text}")
                
                # Create sign images
                sign_images = []
                for letter in filtered_text:
                    img = load_sign_image(letter)
                    sign_images.append(img)
                
                # Display images in a grid
                cols = st.columns(5)
                for i, img in enumerate(sign_images):
                    with cols[i % 5]:
                        st.image(img, caption=f"Sign for '{filtered_text[i]}'", use_column_width=True)
                    
                    if (i + 1) % 5 == 0 and (i + 1) < len(sign_images):
                        cols = st.columns(5)
                
                st.success(f"Converted {len(filtered_text)} letters successfully!")
                #st.balloons()
            else:
                st.warning("No alphabetic characters found in the input text!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
#st.markdown("---")
#st.markdown("""
#<div style='text-align: center'>
#    <p>Built using Streamlit, MediaPipe, and PyTorch</p>
#    <p>Sign Language Recognition System</p>
#</div>
#""", unsafe_allow_html=True)
