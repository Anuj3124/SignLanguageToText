import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import tkinter as tk
from PIL import Image, ImageTk
from utils.sentence_builder import SentenceBuilder

# Load model and label encoder
with open("models/gesture_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
sb = SentenceBuilder()

# OpenCV camera
cap = cv2.VideoCapture(0)
last_pred_time = 0
pred_interval = 1  # Prediction interval in seconds
current_word = ""
confidence_threshold = 0.7

# Tkinter UI
root = tk.Tk()
root.title("Sign Language to Text")
root.geometry("960x700")
root.configure(bg="#121212")

# Webcam frame
video_label = tk.Label(root, bg="#121212")
video_label.pack(pady=(10, 5))

# Sentence and word labels
sentence_var = tk.StringVar()
sentence_label = tk.Label(root, textvariable=sentence_var, font=("Calibri", 18), fg="#00BFFF", bg="#121212")
sentence_label.pack(pady=5)

current_var = tk.StringVar()
current_label = tk.Label(root, textvariable=current_var, font=("Calibri", 14), fg="#FFFFFF", bg="#121212")
current_label.pack(pady=(0, 10))

# Buttons
def reset_sentence():
    sb.reset()
    sentence_var.set("Sentence : ")
    current_var.set("")

def quit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_frame = tk.Frame(root, bg="#121212")
btn_frame.pack(pady=10)

reset_btn = tk.Button(btn_frame, text="RESET", command=reset_sentence,
                      font=("Calibri", 12, "bold"), bg="#1F1F1F", fg="#FFFFFF",
                      padx=20, pady=8, relief="flat")
reset_btn.pack(side=tk.LEFT, padx=20)

quit_btn = tk.Button(btn_frame, text="QUIT", command=quit_app,
                     font=("Calibri", 12, "bold"), bg="#BE0000", fg="#FFFFFF",
                     padx=20, pady=8, relief="flat")
quit_btn.pack(side=tk.RIGHT, padx=20)

# Update loop
def update():
    global last_pred_time, current_word
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            landmarks = [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]

            if len(landmarks) == 63 and (time.time() - last_pred_time) > pred_interval:
                probs = model.predict_proba([landmarks])[0]
                max_prob = np.max(probs)
                pred_index = np.argmax(probs)

                print(f"[INFO] Confidence: {max_prob:.2f}")

                if max_prob > confidence_threshold:
                    current_word = le.inverse_transform([pred_index])[0]
                    sb.add_word(current_word)
                    last_pred_time = time.time()

    # Update sentence and word
    sentence_var.set("Sentence : " + sb.get_sentence())
    current_var.set(f"Current Word : {current_word}")

    # Convert image for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update)

# Run app
update()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
