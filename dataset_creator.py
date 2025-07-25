import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

#set of words
LABELS = []
SAMPLES_PER_LABEL = 100
output = []

#initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

for label in LABELS:
    print(f"\nCollecting data for: {label}")
    count = 0
    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                landmarks = [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:
                    output.append([label] + landmarks)
                    count += 1
                    print(f"Collected {count}/{SAMPLES_PER_LABEL} for '{label}'")
                    time.sleep(0.2)

        cv2.putText(image, f"{label} - {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dataset Collection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#save to CSV
csv_path = "data/gesture_data.csv"
os.makedirs("data", exist_ok=True)

columns = ['label'] + [f'coord_{i}' for i in range(63)]
df_new = pd.DataFrame(output, columns=columns)

if len(output) == 0:
    print("No data was collected. Make sure hand landmarks are detected.")
    exit()

#append to CSV
if os.path.exists(csv_path):
    try:
        df_existing = pd.read_csv(csv_path)
        if list(df_existing.columns) == list(df_new.columns):
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(csv_path, index=False)
            print(f"✅ Appended {len(df_new)} rows to existing file.")
        else:
            print("Column mismatch detected. Data not appended.")
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
else:
    df_new.to_csv(csv_path, index=False)
    print(f"Created new file with {len(df_new)} rows.")
