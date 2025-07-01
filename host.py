"""
Sign Language Recognition System - Complete Version
Webcam fixed for Streamlit Cloud deployment + Mobile responsive
All original UI, game logic, and functionality preserved
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import time
import mediapipe as mp
import torch
import pandas as pdz
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import random
from pathlib import Path

from CNNModel import CNNModel

# Page config - keep exactly as original
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Mobile responsive CSS + original styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
* { font-family: 'Montserrat', sans-serif !important; }
/* Make video responsive for mobile */
video { width: 100% !important; height: auto !important; }
@media (max-width: 480px) {
  .block-container { padding: 0 0.5rem; }
}
.stSelectbox:focus { outline: none !important; box-shadow: none !important; }
@keyframes pulse {
    0% { transform: scale(1); } 50% { transform: scale(1.05); } 100% { transform: scale(1); }
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
.card {
    background: rgba(255, 255, 255, 0.8) !important;
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 25px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border: none;
}
.main-title {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    color: #2c3e50 !important;
    text-align: center;
    margin-bottom: 1rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}
.prediction-badge {
    font-size: 1.5rem;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 30px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# Mediapipe setup - unchanged
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.2)

# Model loading - unchanged
@st.cache_resource
def load_models():
    model_alpha = CNNModel()
    model_alpha.load_state_dict(torch.load("trained.pth", map_location="cpu"))
    model_alpha.eval()
    return model_alpha

model_alpha = load_models()

# Alphabet mapping - unchanged
alphabet_classes = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

# Prediction function - unchanged
def predict_sign_realtime(frame, model, classes_reverse):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    
    coordinates = []
    x_coords, y_coords, z_coords = [], [], []
    predicted_character = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
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

# Sign image loader - unchanged
#def load_sign_image(letter):
    #image_path = f"alphabets/{letter}.jpg"
    #if os.path.exists(image_path):
        #return Image.open(image_path)
def load_sign_image(letter):
    base_path = get_base_path()
    image_path = base_path / "alphabets" / f"{letter}.jpg"
    if image_path.exists():
        return Image.open(str(image_path))
    else:
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

# WebRTC callback for real-time detection
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    # Flip horizontally for mirror effect (like original)
    img_bgr = cv2.flip(img_bgr, 1)
    processed, _ = predict_sign_realtime(img_bgr, model_alpha, alphabet_classes)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Game logic - completely unchanged
def guess_the_character_game():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Guess the Sign Language Character Game")

    # Initialize session state
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'total' not in st.session_state:
        st.session_state.total = 0
    if 'random_letter' not in st.session_state:
        st.session_state.random_letter = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    if 'guess_result' not in st.session_state:
        st.session_state.guess_result = None
    if 'game_locked' not in st.session_state:
        st.session_state.game_locked = False

    # Timer logic
    TIME_LIMIT = 10  # seconds
    elapsed = int(time.time() - st.session_state.start_time)
    time_left = max(0, TIME_LIMIT - elapsed)

    # Auto-refresh every second to update timer
    if not st.session_state.game_locked:
        st_autorefresh(interval=1000, key="timer_refresh")

    # Show scoreboard
    st.info(f"Score: {st.session_state.score} / {st.session_state.total}")
    st.warning(f"Time left: {time_left} seconds")

    # Show image
    letter = st.session_state.random_letter
    img_path = f"alphabets/{letter}.jpg"
    if os.path.exists(img_path):
        st.image(img_path, caption="Which letter is this?", width=300)
    else:
        st.warning("Image not found!")

    # Time up logic
    if time_left <= 0 and not st.session_state.game_locked:
        st.session_state.total += 1
        st.session_state.guess_result = f"Time's up! The correct letter was '{letter}'"
        st.session_state.game_locked = True

    # If game not locked, allow guessing
    if not st.session_state.game_locked:
        guess = st.selectbox("Select your guess:", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if st.button("Submit Guess"):
            st.session_state.total += 1
            if guess == letter:
                st.session_state.score += 1
                st.session_state.guess_result = "Correct!"
            else:
                st.session_state.guess_result = f"Incorrect! It was '{letter}'"
            st.session_state.game_locked = True

    # Show result
    if st.session_state.game_locked and st.session_state.guess_result:
        if "Correct" in st.session_state.guess_result:
            st.success(st.session_state.guess_result)
        elif "Time's up" in st.session_state.guess_result:
            st.warning(st.session_state.guess_result)
        else:
            st.error(st.session_state.guess_result)

        if st.button("Next"):
            st.session_state.random_letter = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            st.session_state.start_time = time.time()
            st.session_state.guess_result = None
            st.session_state.game_locked = False

    st.markdown('</div>', unsafe_allow_html=True)

# Main UI - exactly as original
st.markdown('<p class="main-title">Sign Language Recognition System</p>', unsafe_allow_html=True)

# Sidebar dropdown - unchanged
app_mode = st.sidebar.selectbox(
    "Select Mode:",
    ["Live Detection", "English to Sign Language", "Guess the Character"],
    key="mode_selector"
)

# Routing with webcam fix
if app_mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Sign Language Detection")
    
    # WebRTC for real-time detection (works on mobile)
    st.subheader("Real-time Detection (Browser Camera)")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    webrtc_ctx = webrtc_streamer(
        key="sign-detection",
        video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("📱 Works on desktop and mobile browsers. Grant camera permission when prompted.")
    
    # Fallback: Single photo capture for mobile
    st.subheader("Alternative: Photo Capture (Mobile Friendly)")
    with st.expander("Use if real-time detection doesn't work on your device"):
        img_file = st.camera_input("Take a photo of your sign")
        if img_file:
            img = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)  # Mirror effect
            processed, prediction = predict_sign_realtime(frame, model_alpha, alphabet_classes)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(processed, caption="Processed Image", use_column_width=True)
            with col2:
                if prediction:
                    st.markdown(f'<div class="prediction-badge" style="background-color: #4CAF50; color: white;">Detected: {prediction}</div>', unsafe_allow_html=True)
                else:
                    st.info("No sign detected")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "English to Sign Language":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("English to Sign Language Converter")
    user_text = st.text_input("Enter text to convert to sign language:", placeholder="Type your message here...", help="Only alphabetic characters will be converted")
    if st.button("Convert to Sign Language", use_container_width=True):
        if not user_text:
            st.error("Please enter some text to convert!")
        else:
            filtered_text = ''.join([char.upper() for char in user_text if char.isalpha()])
            if filtered_text:
                st.subheader(f"Sign Language for: '{user_text}'")
                st.caption(f"Filtered letters: {filtered_text}")
                sign_images = [load_sign_image(letter) for letter in filtered_text]
                cols = st.columns(5)
                for i, img in enumerate(sign_images):
                    with cols[i % 5]:
                        st.image(img, caption=f"Sign for '{filtered_text[i]}'", use_column_width=True)
                    if (i + 1) % 5 == 0 and (i + 1) < len(sign_images):
                        cols = st.columns(5)
                st.success(f"Converted {len(filtered_text)} letters successfully!")
            else:
                st.warning("No alphabetic characters found in the input text!")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Guess the Character":
    guess_the_character_game()
