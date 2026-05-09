from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import pyttsx3

# =======================
# Flask App
# =======================
app = Flask(__name__)

tts_engine = pyttsx3.init()


# =======================
# Word recommendation list
# =======================
VOCABULARY = [
    "hello", "hey", "help", "how", "house",
    "hi", "happy", "hand",
    "yes", "you", "your",
    "thank", "thanks", "there",
    "good", "great", "go",
    "please", "people", "practice"
]


# =======================
# Load trained model
# =======================
model = tf.keras.models.load_model("asl_model_final.keras")
labels = [chr(i) for i in range(65, 91)]

# =======================
# Configuration
# =======================
IMG_SIZE = 128
CONF_THRESHOLD = 0.7
SMOOTH_FRAMES = 8
ROI_SIZE = 220

# =======================
# Prediction smoothing
# =======================
pred_queue = deque(maxlen=SMOOTH_FRAMES)

# =======================
# Word formation state
# =======================
current_word = ""
current_prediction = ""
last_letter = None
stable_count = 0
gap_counter = 0
capture_progress = 0

LETTER_STABILITY_FRAMES = 12
SPACE_GAP_FRAMES = 20

# =======================
# Webcam
# =======================
cap = cv2.VideoCapture(0)

# =======================
# Video generator
# =======================
def generate_frames():
    global current_word, last_letter, stable_count, gap_counter, current_prediction, capture_progress

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # =======================
        # ROI AT BOTTOM-LEFT
        # =======================
        x1, y2 = 20, h - 20
        x2, y1 = x1 + ROI_SIZE, y2 - ROI_SIZE
        roi = frame[y1:y2, x1:x2]

        display_text = "Place hand inside box"
        color = (200, 200, 200)

        if roi.size != 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (IMG_SIZE, IMG_SIZE))
            roi_norm = roi_resized / 255.0
            roi_input = np.expand_dims(roi_norm, axis=0)

            pred = model.predict(roi_input, verbose=0)[0]
            pred_queue.append(pred)

            avg_pred = np.mean(pred_queue, axis=0)
            confidence = np.max(avg_pred)
            letter = labels[np.argmax(avg_pred)]
            current_prediction = letter

            if confidence >= CONF_THRESHOLD:
                display_text = f"{letter} ({confidence*100:.1f}%)"
                color = (0, 255, 0)

                gap_counter = 0

                if letter == last_letter:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_letter = letter
                capture_progress = min(
                    int((stable_count / LETTER_STABILITY_FRAMES) * 100),
                    100
                )

                if capture_progress > 100:
                    capture_progress = 100
                    
                if stable_count == LETTER_STABILITY_FRAMES:
                    current_word += letter

            else:
                display_text = "Low confidence"
                color = (0, 165, 255)
                stable_count = 0
                
                # gap_counter += 1

                if gap_counter == SPACE_GAP_FRAMES and len(current_word) > 0:
                    current_word += " "

        # Drawing
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand inside box",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (255, 255, 255), 2)
        

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

def get_recommendations(word):
    if not word:
        return []

    prefix = word.strip().split(" ")[-1].lower()
    if len(prefix) < 2:
        return []

    matches = [w for w in VOCABULARY if w.startswith(prefix)]
    return matches[:5]

# =======================
# Routes
# =======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# 🔄 Clear word
@app.route("/clear", methods=["POST"])
def clear_word():
    global current_word
    current_word = ""
    
    
    return ("", 204)

# ⌫ Backspace
@app.route("/backspace", methods=["POST"])
def backspace():
    global current_word
    current_word = current_word[:-1]
    return ("", 204)

# ⎵ Space
@app.route("/space", methods=["POST"])
def space():
    global current_word
    current_word += " "
    return ("", 204)

@app.route("/recommendations")
def recommendations():
    recs = get_recommendations(current_word)
    return {"suggestions": recs}


@app.route("/get_text")
def get_text():
    return {
        "current_char": current_prediction,
        "sentence": current_word,
        "progress": capture_progress
    }
    
@app.route("/apply_suggestion", methods=["POST"])
def apply_suggestion():
    global current_word
    data = request.json
    suggestion = data.get("word", "")

    parts = current_word.strip().split(" ")
    parts[-1] = suggestion
    current_word = " ".join(parts) + " "

    return ("", 204)

@app.route("/speak", methods=["POST"])
def speak():
    if current_word.strip():
        tts_engine.say(current_word)
        tts_engine.runAndWait()
    return ("", 204)



# =======================
# Entry point
# =======================
if __name__ == "__main__":
    app.run(debug=False)
