import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import time

# Define labels and sample size
LABELS = ['smile']  # Add more labels as needed
SAMPLES_PER_LABEL = 100
output = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Collect data for each label
for label in LABELS:
    print(f"Collecting data for: {label}")
    count = 0
    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                landmarks = [coord for lm in hand.landmark for coord in (lm.x, lm.y, lm.z)]
                if len(landmarks) == 63:  # Ensure valid hand data
                    output.append([label] + landmarks)
                    count += 1
                    time.sleep(0.2)

        # Show progress
        cv2.putText(image, f"{label} - {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Dataset Collection", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
# Create output folder if it doesn't exist
os.makedirs("data", exist_ok=True)
csv_path = "data/gesture_data.csv"

# Define proper column names
columns = ['label'] + [f'{axis}{i}' for i in range(21) for axis in 'xyz']

# Convert new samples to DataFrame with correct column names
new_df = pd.DataFrame(output, columns=columns)

# If file exists, read old data and append safely
if os.path.exists(csv_path):
    old_df = pd.read_csv(csv_path)
    old_df.columns = columns  # enforce same headers

    # Now concat safely
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    print(f"\n✅ Data appended and saved to {csv_path}")
else:
    new_df.to_csv(csv_path, index=False)
    print(f"\n✅ New file created and saved to {csv_path}")