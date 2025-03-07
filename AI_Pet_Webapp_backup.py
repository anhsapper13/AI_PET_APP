import streamlit as st
import cv2
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from fer import FER
import time
import face_recognition
import os
import json
from datetime import datetime

# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create directories for storing data
os.makedirs("data", exist_ok=True)
os.makedirs("data/user", exist_ok=True)
os.makedirs("data/session", exist_ok=True)

# ======= 1. AUTHENTICATION MODULE =======

class UserAuth:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
    def capture_and_save_owner_embedding(self, webcam_placeholder):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        
        start_time = time.time()
        timeout = 10  # Timeout after 10 seconds
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            pil_img = Image.fromarray(frame_rgb)
            boxes, _ = self.mtcnn.detect(pil_img)
            
            if boxes is not None:
                areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
                largest_face_index = areas.index(max(areas))
                largest_face = self.mtcnn.extract(pil_img, [boxes[largest_face_index]], None)
                
                if largest_face is not None:
                    owner_embedding = self.resnet(largest_face).detach()
                    
                    # Save both embedding and user info
                    user_data = {
                        "username": "Owner",
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    torch.save(owner_embedding, 'data/user/facial_embedding.pt')
                    with open('data/user/user_info.json', 'w') as f:
                        json.dump(user_data, f)
                        
                    cap.release()
                    return True
        
        cap.release()
        return False
    
    def is_owner_present(self):
        if not os.path.exists('data/user/facial_embedding.pt'):
            return False

        try:
            owner_embedding = torch.load('data/user/facial_embedding.pt')
        except:
            return False

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        owner_detected = False
        start_time = time.time()
        
        webcam_placeholder = st.empty()
        status_placeholder = st.empty()
        status_placeholder.text("Looking for your face...")

        try:
            while time.time() - start_time < 5:  # Check for 5 seconds
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                webcam_placeholder.image(frame_rgb, caption='Checking if I know you...', channels='RGB', use_container_width=True)
                
                pil_img = Image.fromarray(frame_rgb)
                boxes, _ = self.mtcnn.detect(pil_img)
                
                if boxes is not None:
                    faces = self.mtcnn.extract(pil_img, boxes, None)
                    if faces is not None:
                        if not isinstance(faces, list):
                            faces = [faces]
                        
                        for face in faces:
                            if face.dim() == 4:
                                face = face[0]
                            current_embedding = self.resnet(face.unsqueeze(0)).detach()
                            distance = (current_embedding - owner_embedding).norm().item()
                            if distance < 0.8:  # Increased threshold for better recognition
                                owner_detected = True
                                break

                if owner_detected:
                    status_placeholder.success("I remember you! 🐾")
                    break

        finally:
            cap.release()
            webcam_placeholder.empty()

        return owner_detected

# ======= 2. CHAT MODULE =======

class ChatHandler:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
        self.chat_log_path = "data/session/chat_logs.json"
        
    def get_response(self, user_input):
        if user_input.lower() == 'quit':
            return "Woof woof! Bye!"

        input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
        chat_output_ids = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id, num_return_sequences=1)
        bot_output = self.tokenizer.decode(chat_output_ids[0], skip_special_tokens=True)

        if user_input.lower() in bot_output.lower():
            bot_output = bot_output[len(user_input):].strip()
            
        # Log the chat
        self.log_chat(user_input, bot_output)
            
        return bot_output
    
    def log_chat(self, user_input, bot_output):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if os.path.exists(self.chat_log_path):
            with open(self.chat_log_path, 'r') as f:
                try:
                    chat_logs = json.load(f)
                except:
                    chat_logs = []
        else:
            chat_logs = []
            
        chat_logs.append({
            "timestamp": current_time,
            "user": user_input,
            "bot": bot_output
        })
        
        with open(self.chat_log_path, 'w') as f:
            json.dump(chat_logs, f, indent=2)

# ======= 3. EMOTION ANALYSIS MODULE =======

class EmotionAnalyzer:
    def __init__(self):
        self.detector = FER(mtcnn=True)
        self.emotion_log_path = "data/session/emotion_logs.json"
        
    def detect_emotion(self):
        emotion = self.continuous_emotion_detection()
        
        # Log the emotion
        self.log_emotion(emotion)
        
        # Define responses based on the dominant emotion
        responses = {
            "sad": "Why are you sad? What happened? I'll try to cheer you up! *nuzzles*",
            "angry": "Oh no, you seem angry! *ears down* Want to talk about it?",
            "happy": "Woof Woof! *tail wagging* I'm glad to see you happy!",
            "neutral": "You seem calm today! *sits attentively*",
            "fear": "Don't be afraid! I'm here to protect you! *protective stance*",
            "surprise": "What's got you surprised? *tilts head curiously*",
            "disgust": "Something bothering you? *concerned look*"
        }

        if emotion in responses:
            return responses[emotion]
        else:
            return f"I see you're feeling something... *watches attentively* Want to tell me about it?"
    
    def continuous_emotion_detection(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        emotions = []
        emotions_detail = []  # For debugging
        
        try:
            with st.spinner("Reading your emotions..."):
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret:
                        result = self.detector.detect_emotions(frame)
                        
                        if result and len(result) > 0:
                            emotions_dict = result[0]['emotions']
                            emotions_detail.append(emotions_dict)
                            dominant_emotion = max(emotions_dict.items(), key=lambda x: x[1])[0]
                            emotions.append(dominant_emotion)
                            
                    time.sleep(0.1)

                if emotions:
                    emotion_counts = {emotion: emotions.count(emotion) for emotion in set(emotions)}
                    # For debugging:
                    st.write("Emotion detection results:", emotion_counts)
                    
                    # Average the emotion scores across frames
                    avg_emotions = {}
                    for emotion_dict in emotions_detail:
                        for emotion, score in emotion_dict.items():
                            if emotion not in avg_emotions:
                                avg_emotions[emotion] = []
                            avg_emotions[emotion].append(score)
                    
                    for emotion in avg_emotions:
                        avg_emotions[emotion] = sum(avg_emotions[emotion]) / len(avg_emotions[emotion])
                    
                    # Get the emotion with highest average score
                    dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0]
                    return dominant_emotion
                return "neutral"
        finally:
            cap.release()
    
    def log_emotion(self, emotion):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if os.path.exists(self.emotion_log_path):
            with open(self.emotion_log_path, 'r') as f:
                try:
                    emotion_logs = json.load(f)
                except:
                    emotion_logs = []
        else:
            emotion_logs = []
            
        emotion_logs.append({
            "timestamp": current_time,
            "emotion": emotion
        })
        
        with open(self.emotion_log_path, 'w') as f:
            json.dump(emotion_logs, f, indent=2)

# ======= MAIN APPLICATION =======

def main_page():
    st.markdown("# Woof Woof! I've Been Waiting to Play with You!")
    st.image("dog1.gif", use_container_width=True)
    
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if "last_emotion_check" not in st.session_state:
        st.session_state["last_emotion_check"] = 0

    # Initialize modules
    chat_handler = ChatHandler()
    emotion_analyzer = EmotionAnalyzer()

    # Create columns for the layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        prompt_response = st.empty()
        user_input = st.text_input("Type your message here:", key="user_input")
        send_button = st.button("Send", key="send_button")
    
    with col2:
        check_emotion = st.button("🎭 Check My Mood", key="check_emotion")

    if check_emotion:
        emotion_response = emotion_analyzer.detect_emotion()
        prompt_response.markdown(f"**Pet's Response:** {emotion_response}")
        st.session_state["chat_history"].append(("🐶", emotion_response))

    if send_button and user_input:
        bot_output = chat_handler.get_response(user_input)
        prompt_response.markdown(f"**Pet's Response:** {bot_output}")
        st.session_state["chat_history"].append(("👶", user_input))
        st.session_state["chat_history"].append(("🐶", bot_output))
        
        # Periodic emotion check
        if time.time() - st.session_state["last_emotion_check"] > 30:
            emotion_response = emotion_analyzer.detect_emotion()
            prompt_response.markdown(f"**Pet's Response:** {emotion_response}")
            st.session_state["chat_history"].append(("🐶", emotion_response))
            st.session_state["last_emotion_check"] = time.time()

    # Sidebar for chat history
    st.sidebar.markdown("# Chat History")
    for speaker, message in st.session_state["chat_history"]:
        st.sidebar.text(f"{speaker}: {message}")

def main():
    # Initialize the authentication module
    user_auth = UserAuth()
    
    # Check if the owner embedding file exists
    owner_file_exists = os.path.exists('data/user/facial_embedding.pt')
    
    # Reset recognition if no embedding file exists
    if not owner_file_exists:
        st.session_state["owner_present_checked"] = False
        st.session_state["owner_recognized"] = False
    
    # Initialize session states if they don't exist
    if "owner_present_checked" not in st.session_state:
        st.session_state["owner_present_checked"] = False
    if "owner_recognized" not in st.session_state:
        st.session_state["owner_recognized"] = False
    
    # If owner is already recognized, go straight to chat
    if st.session_state["owner_recognized"]:
        main_page()
        return
    
    # If not yet checked and embedding exists, try to recognize
    if not st.session_state["owner_present_checked"] and owner_file_exists:
        owner_present = user_auth.is_owner_present()
        st.session_state["owner_present_checked"] = True
        st.session_state["owner_recognized"] = owner_present
        if owner_present:
            main_page()
            return
    
    # If not recognized, show the registration page
    st.markdown("# Hello! I'm your AI Pet!")
    st.markdown("### I need to get to know you first before we can chat.")
    
    if "capture_in_progress" not in st.session_state:
        st.session_state["capture_in_progress"] = False

    if st.button("Get to Know Me", key="get_to_know"):
        st.session_state["capture_in_progress"] = True

    if st.session_state["capture_in_progress"]:
        st.markdown("### 📸 Looking for your face...")
        st.markdown("Please look at the camera and stay still.")
        
        webcam_placeholder = st.empty()
        
        try:
            success = user_auth.capture_and_save_owner_embedding(webcam_placeholder)
            if success:
                st.session_state["capture_in_progress"] = False
                st.session_state["owner_present_checked"] = True
                st.session_state["owner_recognized"] = True
                st.success("✅ Got it! I'll remember you now!")
                time.sleep(2)
                st.experimental_rerun()
            else:
                st.error("❌ I couldn't see your face clearly. Please try again.")
                st.session_state["capture_in_progress"] = False
        except Exception as e:
            st.error(f"❌ Something went wrong: {str(e)}")
            st.session_state["capture_in_progress"] = False
    else:
        st.markdown("# Sorry, I don't talk to strangers.")
        st.image("notouch.png", use_container_width=True)

if __name__ == "__main__":
    main()
