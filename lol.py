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
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import random
from pathlib import Path
from CNNModel import CNNModel

# ───────────────────────── Path helper ─────────────────────────
def get_base_path() -> Path:
    """Return the directory where this script resides."""
    return Path(__file__).parent

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
    model_alpha.load_state_dict(torch.load(str(get_base_path() / "trained.pth"), map_location="cpu"))
    model_alpha.eval()
    return model_alpha

model_alpha = load_models()

# Alphabet mapping - unchanged
alphabet_classes = {i: chr(65 + i) for i in range(26)}

# Prediction function - unchanged
def predict_sign_realtime(frame, model, classes_reverse):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    
    x_coords, y_coords, z_coords = [], [], []
    predicted_character = None
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            data = {}
            for i, lm in enumerate(hand_landmarks.landmark):
                x_coords.append(lm.x)
                y_coords.append(lm.y)
                z_coords.append(lm.z)
            for i, landmark in enumerate(mp_hands.HandLandmark):
                lm = hand_landmarks.landmark[i]
                data[f"{landmark.name}_x"] = lm.x - min(x_coords)
                data[f"{landmark.name}_y"] = lm.y - min(y_coords)
                data[f"{landmark.name}_z"] = lm.z - min(z_coords)
            
            coords_array = np.reshape(np.array(list(data.values())), (1, 63, 1))
            coords_tensor = torch.from_numpy(coords_array).float()
            with torch.no_grad():
                _, pred = torch.max(model(coords_tensor).data, 1)
            predicted_character = classes_reverse[pred.item()]
            
            h, w, _ = frame.shape
            x1, y1 = int(min(x_coords) * w) - 10, int(min(y_coords) * h) - 10
            x2, y2 = int(max(x_coords) * w) + 10, int(max(y_coords) * h) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            return frame, predicted_character
    return frame, None

# Sign image loader with fixed path handling
def load_sign_image(letter):
    base_path = get_base_path()
    image_path = base_path / "alphabets" / f"{letter}.jpg"
    if image_path.exists():
        return Image.open(str(image_path))
    # Fallback – draw letter
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()
    tw = draw.textlength(letter, font=font)
    draw.text(((200 - tw) / 2, 50), letter, fill="black", font=font)
    return img

# WebRTC callback for real-time detection
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img_bgr = frame.to_ndarray(format="bgr24")
    img_bgr = cv2.flip(img_bgr, 1)  # Mirror effect
    processed, _ = predict_sign_realtime(img_bgr, model_alpha, alphabet_classes)
    return av.VideoFrame.from_ndarray(processed, format="bgr24")

# Game logic - unchanged
def guess_the_character_game():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Guess the Sign Language Character Game")
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "total" not in st.session_state:
        st.session_state.total = 0
    if "random_letter" not in st.session_state:
        st.session_state.random_letter = random.choice(list(alphabet_classes.values()))
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "guess_result" not in st.session_state:
        st.session_state.guess_result = None
    if "game_locked" not in st.session_state:
        st.session_state.game_locked = False

    TIME_LIMIT = 10
    elapsed = int(time.time() - st.session_state.start_time)
    time_left = max(0, TIME_LIMIT - elapsed)
    if not st.session_state.game_locked:
        st_autorefresh(interval=1000, key="timer_refresh")

    st.info(f"Score: {st.session_state.score} / {st.session_state.total}")
    st.warning(f"Time left: {time_left} seconds")

    letter = st.session_state.random_letter
    sign_img = load_sign_image(letter)
    st.image(sign_img, caption="Which letter is this?", width=300)

    if time_left <= 0 and not st.session_state.game_locked:
        st.session_state.total += 1
        st.session_state.guess_result = f"Time's up! The correct letter was '{letter}'"
        st.session_state.game_locked = True

    if not st.session_state.game_locked:
        guess = st.selectbox("Select your guess:", list(alphabet_classes.values()))
        if st.button("Submit Guess"):
            st.session_state.total += 1
            if guess == letter:
                st.session_state.score += 1
                st.session_state.guess_result = "Correct!"
            else:
                st.session_state.guess_result = f"Incorrect! It was '{letter}'"
            st.session_state.game_locked = True

    if st.session_state.game_locked and st.session_state.guess_result:
        if "Correct" in st.session_state.guess_result:
            st.success(st.session_state.guess_result)
        elif "Time's up" in st.session_state.guess_result:
            st.warning(st.session_state.guess_result)
        else:
            st.error(st.session_state.guess_result)
        if st.button("Next"):
            st.session_state.random_letter = random.choice(list(alphabet_classes.values()))
            st.session_state.start_time = time.time()
            st.session_state.guess_result = None
            st.session_state.game_locked = False

    st.markdown('</div>', unsafe_allow_html=True)

# Main UI - exactly as original
st.markdown('<p class="main-title">Sign Language Recognition System</p>', unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Select Mode:",
    ["Live Detection", "English to Sign Language", "Guess the Character"],
    key="mode_selector"
)

if app_mode == "Live Detection":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live Sign Language Detection")

    st.subheader("Real-time Detection (Browser Camera)")
    rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    webrtc_streamer(
        key="sign-detection",
        video_frame_callback=video_frame_callback,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.caption("📱 Works on desktop and mobile browsers. Grant camera permission when prompted.")

    st.subheader("Alternative: Photo Capture (Mobile Friendly)")
    with st.expander("Use if real-time detection doesn't work on your device"):
        img_file = st.camera_input("Take a photo of your sign")
        if img_file:
            img = Image.open(img_file).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            frame = cv2.flip(frame, 1)
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
            filtered = ''.join([c.upper() for c in user_text if c.isalpha()])
            if filtered:
                st.subheader(f"Sign Language for: '{user_text}'")
                st.caption(f"Filtered letters: {filtered}")
                cols = st.columns(5)
                for i, ch in enumerate(filtered):
                    img = load_sign_image(ch)
                    with cols[i % 5]:
                        st.image(img, caption=f"Sign for '{ch}'", use_column_width=True)
                    if (i + 1) % 5 == 0 and (i + 1) < len(filtered):
                        cols = st.columns(5)
                st.success(f"Converted {len(filtered)} letters successfully!")
            else:
                st.warning("No alphabetic characters found!")
    st.markdown('</div>', unsafe_allow_html=True)

elif app_mode == "Guess the Character":
    guess_the_character_game()
