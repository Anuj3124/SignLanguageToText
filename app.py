import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from utils.sentence_builder import SentenceBuilder

#load model
with open("models/gesture_model.pkl", "rb") as f:
    model, le = pickle.load(f)

#mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
sb = SentenceBuilder()

#opencv
cap = cv2.VideoCapture(0)
last_pred_time = 0
pred_interval = 2
current_word = ""

#tkinter ui
root = tk.Tk()
root.title("Sign Language to Text Converter")
root.configure(bg="#121212")
root.geometry("960x650")

#fonts and colours
FONT_MAIN = ("Calibri", 18)
FONT_SUB = ("Calibri", 14)
FONT_BTN = ("Calibri", 12, "bold")
COLOR_BG = "#121212"
COLOR_TEXT = "#FFFFFF"
COLOR_ACCENT = "#00BFFF"
COLOR_BTN = "#1F1F1F"
COLOR_BTN_HOVER = "#2D2D2D"
COLOR_BTN_ACTIVE = "#3A3A3A"
COLOR_QUIT = "#BE0000"
COLOR_QUIT_HOVER = "#770000"
COLOR_QUIT_ACTIVE = "#990000"

#hover effect
def add_hover_effect(button, bg_normal, bg_hover, bg_active):
    def on_enter(e):
        button['background'] = bg_hover
    def on_leave(e):
        button['background'] = bg_normal
    def on_press(e):
        button['background'] = bg_active
    def on_release(e):
        button['background'] = bg_hover
    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)
    button.bind("<ButtonPress-1>", on_press)
    button.bind("<ButtonRelease-1>", on_release)

#webcam
video_label = tk.Label(root, bg=COLOR_BG)
video_label.pack(pady=(10, 5))

#sentence
sentence_var = tk.StringVar()
sentence_label = tk.Label(root, textvariable=sentence_var, font=FONT_MAIN,
                          fg=COLOR_ACCENT, bg=COLOR_BG)
sentence_label.pack(pady=5)

#current word
current_var = tk.StringVar()
current_label = tk.Label(root, textvariable=current_var, font=FONT_SUB,
                         fg=COLOR_TEXT, bg=COLOR_BG)
current_label.pack(pady=(0, 10))

#loop to update
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
                pred = model.predict([landmarks])[0]
                current_word = le.inverse_transform([pred])[0]
                sb.add_word(current_word)
                last_pred_time = time.time()

    sentence_var.set("Sentence : " + sb.get_sentence())
    current_var.set(f"Current Word : {current_word}")

    #tkinter image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update)

#button fns
def reset_sentence():
    sb.reset()
    sentence_var.set("Sentence : ")
    current_var.set("")

def quit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

#buttons
btn_frame = tk.Frame(root, bg=COLOR_BG)
btn_frame.pack(pady=20)

reset_btn = tk.Button(btn_frame, text="RESET", font=FONT_BTN,
                      bg=COLOR_BTN, fg=COLOR_TEXT, relief="flat",
                      padx=20, pady=8, command=reset_sentence)
reset_btn.pack(side=tk.LEFT, padx=40)
add_hover_effect(reset_btn, COLOR_BTN, COLOR_BTN_HOVER, COLOR_BTN_ACTIVE)

quit_btn = tk.Button(btn_frame, text="QUIT", font=FONT_BTN,
                     bg=COLOR_QUIT, fg="#FFFFFF", relief="flat",
                     padx=20, pady=8, command=quit_app)
quit_btn.pack(side=tk.RIGHT, padx=40)
add_hover_effect(quit_btn, COLOR_QUIT, COLOR_QUIT_HOVER, COLOR_QUIT_ACTIVE)

#run
update()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
