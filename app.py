import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, Response, request, redirect, url_for

# --- Flask App Initialization ---
app = Flask(__name__, template_folder='login') # Point to the 'login' folder for templates

# --- TensorFlow Model Loading ---
# Custom DepthwiseConv2D layer that handles the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

model = None
labels = []

def load_model_and_labels():
    global model, labels
    try:
        model_path = os.path.join("Model", "keras_model.h5")
        labels_path = os.path.join("Model", "labels.txt")

        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            print("Error: Model or labels file not found")
            return False

        model = load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            labels = f.read().splitlines()
        print("Model and labels loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def preprocess_image(img):
    # Ensure image is 3 channels
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0)
    img_normalized = img_array / 255.0
    return img_normalized


# --- Gesture Recognition Logic ---
def generate_frames():
    cap = cv2.VideoCapture(0)

    # --- Try setting a specific resolution ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # --- (Optional) You could also try setting FPS, but it's not always respected ---
    # cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera")
        # Yield a placeholder frame indicating the error
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8) # Standard webcam size
            cv2.putText(frame, "Camera Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1) # Avoid busy loop

    offset = 20
    # --- Reduce intermediate image size ---
    imgSize = 250 # Changed from 300
    # --- Adjust detector confidence and maxHands ---
    detector = HandDetector(detectionCon=0.4, maxHands=1) # Lowered detectionCon
    current_prediction = "" # Variable to hold the current prediction text
    last_bbox = None # Store the last known bounding box
    frame_count = 0 # Initialize frame counter
    # --- Process even fewer frames ---
    process_every_n_frames = 4 # Changed from 2 to 4

    if model is None or not labels:
        print("Model not loaded, cannot perform prediction.")
        # Yield a placeholder frame indicating the error
        while True:
            frame = np.zeros((480, 640, 3), dtype=np.uint8) # Standard webcam size
            cv2.putText(frame, "Model Loading Error", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1) # Avoid busy loop

    print("Starting video stream...")
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read frame")
            time.sleep(0.1)
            continue

        imgOutput = img.copy()
        frame_count += 1
        prediction_made_this_cycle = False # Reset this flag

        # --- Only process every Nth frame ---
        if frame_count % process_every_n_frames == 0:
            hands, _ = detector.findHands(img, draw=False)
            prediction_made_in_processing = False # Reset this flag

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                last_bbox = (x, y, w, h) # Update last known bbox

                # --- Restore prediction code ---
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size > 0:
                    aspectRatio = h / w if w > 0 else 0
                    try:
                        # (Keep the existing resizing logic for imgWhite)
                        if aspectRatio > 1:
                            k = imgSize / h if h > 0 else 0
                            wCal = math.ceil(k * w)
                            if wCal > 0 and imgSize > 0:
                                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                wGap = math.ceil((imgSize - wCal) / 2)
                                imgWhite[:, wGap:min(imgSize, wGap + wCal)] = imgResize[:, :min(wCal, imgSize-wGap)]
                        else:
                            k = imgSize / w if w > 0 else 0
                            hCal = math.ceil(k * h)
                            if hCal > 0 and imgSize > 0:
                                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                hGap = math.ceil((imgSize - hCal) / 2)
                                imgWhite[hGap:min(imgSize, hGap + hCal), :] = imgResize[:min(hCal, imgSize-hGap), :]

                        imgForPrediction = preprocess_image(imgWhite)
                        predictions = model.predict(imgForPrediction, verbose=0) # Prediction is back
                        index = np.argmax(predictions)
                        confidence = predictions[0][index]

                        # Using original confidence threshold for prediction display
                        if confidence > 0.7:
                            label = labels[index]
                            current_prediction = f"Detected: {label} ({confidence:.1%})"
                            prediction_made_in_processing = True
                            prediction_made_this_cycle = True # Mark that prediction happened
                        # else: # Decide if you want to clear prediction immediately on low confidence
                            # current_prediction = ""
                            # last_bbox = None

                    except Exception as e:
                        print(f"Error processing hand: {e}")
                        # current_prediction = "" # Clear prediction on error
                        # last_bbox = None
                # --- End of restored prediction code ---

            # If no hands were detected *in this processing frame*, clear prediction and bbox
            # This logic might need adjustment depending on desired behavior when hand is temporarily lost
            if not prediction_made_in_processing:
                 current_prediction = ""
                 last_bbox = None # Clear the bounding box if no hand is detected in this processing cycle

        # --- Drawing happens on every frame, using the latest results ---
        # Draw bounding box if one was found in the last processed frame
        if last_bbox:
            x, y, w, h = last_bbox
            # Recalculate safe offsets for drawing on the *current* imgOutput dimensions
            y1_draw, y2_draw = max(0, y - offset), min(imgOutput.shape[0], y + h + offset)
            x1_draw, x2_draw = max(0, x - offset), min(imgOutput.shape[1], x + w + offset)
            cv2.rectangle(imgOutput, (x1_draw, y1_draw), (x2_draw, y2_draw), (255, 0, 255), 2)


        # Draw the prediction text (either the latest prediction or "Show a sign...")
        text_to_display = current_prediction if current_prediction else "Show a sign..."
        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)
        padding = 10
        rect_y1 = padding
        rect_y2 = padding + text_height + padding + baseline
        rect_x1 = padding
        rect_x2 = padding + text_width + padding

        # Ensure rectangle coordinates are within image bounds before drawing
        rect_y1, rect_y2 = max(0, rect_y1), min(imgOutput.shape[0], rect_y2)
        rect_x1, rect_x2 = max(0, rect_x1), min(imgOutput.shape[1], rect_x2)

        if rect_y2 > rect_y1 and rect_x2 > rect_x1: # Check if rectangle has valid dimensions
            sub_img = imgOutput[rect_y1:rect_y2, rect_x1:rect_x2]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 50
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            imgOutput[rect_y1:rect_y2, rect_x1:rect_x2] = res

            # Position text inside the potentially clipped rectangle bounds
            text_x = rect_x1 + padding // 2
            text_y = rect_y1 + text_height + padding // 2
            cv2.putText(imgOutput, text_to_display, (text_x, text_y),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)


        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', imgOutput)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    print("Video stream stopped.")
    cap.release()


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the dedicated video viewer page as the main page."""
    # This line makes the main page show the viewer.html content
    return render_template('viewer.html')

# The /viewer route still exists but now serves the same content as /
@app.route('/viewer')
def viewer():
    """Serves the dedicated video viewer page."""
    return render_template('viewer.html')

@app.route('/login', methods=['POST'])
def login():
    """Handles the login form submission."""
    email = request.form.get('email') # Assuming input name is 'email'
    password = request.form.get('password') # Assuming input name is 'password'
    print(f"Login attempt: Email={email}, Password={'*' * len(password) if password else 'None'}")
    # In a real app, you'd verify credentials here
    # For now, just redirect back to index or to a 'logged in' page
    # return redirect(url_for('index'))
    # Or render a success/dashboard page
    return "Login attempt received (check console). <a href='/'>Go back</a>"


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    if model is None or not labels:
        return "Error: Model not loaded.", 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Main Execution ---
if __name__ == "__main__":
    if load_model_and_labels():
        print("Starting Flask server...")
        # Use host='0.0.0.0' to make it accessible on your network
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Server cannot start.")
        