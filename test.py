import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
import os

# Custom DepthwiseConv2D layer that handles the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Register the custom layer
tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

def load_model_and_labels():
    try:
        model_path = os.path.join("Model", "keras_model.h5")
        labels_path = os.path.join("Model", "labels.txt")
        
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Model or labels file not found")
            
        model = load_model(model_path, compile=False)
        with open(labels_path, "r") as f:
            labels = f.read().splitlines()
        return model, labels
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def main():
    # Initialize camera and hand detector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    offset = 20
    imgSize = 300
    detector = HandDetector(maxHands=1)
    
    # Load model and labels
    model, labels = load_model_and_labels()
    if model is None or labels is None:
        return
        
    print("Model loaded successfully. Press 'q' to quit.")
    
    while True:
        success, img = cap.read()
        if not success:
            continue
            
        # Create a copy of the image for the main display
        imgMain = img.copy()
        
        # Detect hands with visualization
        hands, img = detector.findHands(img, draw=False)  # Set draw=False to remove visualization from main feed
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Create white background
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)

            if w > 0 and h > 0:
                try:
                    # Check if the crop coordinates are valid
                    if y-offset >= 0 and x-offset >= 0 and y+h+offset <= img.shape[0] and x+w+offset <= img.shape[1]:
                        imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
                        
                        # Check if imgCrop is not empty
                        if imgCrop.size > 0:
                            aspectRatio = h/w
                            if aspectRatio > 1:
                                k = imgSize/h
                                wCal = math.ceil(k*w)
                                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                wGap = math.ceil((imgSize-wCal)/2)
                                # Ensure the resized image fits within the white background
                                wCal = min(wCal, imgSize)
                                imgWhite[:, wGap:wGap+wCal] = imgResize
                                # Store coordinates for rectangle
                                rectX1, rectY1 = wGap, 0
                                rectX2, rectY2 = wGap + wCal, imgSize
                            else:
                                k = imgSize/w
                                hCal = math.ceil(k*h)
                                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                hGap = math.ceil((imgSize-hCal)/2)
                                # Ensure the resized image fits within the white background
                                hCal = min(hCal, imgSize)
                                imgWhite[hGap:hGap+hCal, :] = imgResize
                                # Store coordinates for rectangle
                                rectX1, rectY1 = 0, hGap
                                rectX2, rectY2 = imgSize, hGap + hCal

                            # Prepare image for prediction
                            imgForPrediction = preprocess_image(imgWhite)
                            
                            # Get prediction
                            predictions = model.predict(imgForPrediction)
                            index = np.argmax(predictions)
                            confidence = predictions[0][index]
                            
                            # Only show predictions with confidence > 50%
                            if confidence > 0.5:
                                label = labels[index]
                                # Display only the alphabet without coordinates
                                cv2.putText(imgMain, label, 
                                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                          2, (0, 255, 0), 2)

                            # Display the processed images with visualization
                            # Create a copy of the original image with hand detection
                            imgWithHands = img.copy()
                            hands, imgWithHands = detector.findHands(imgWithHands)  # Add visualization to original image
                            imgCropWithHands = imgWithHands[y-offset:y + h+offset, x-offset:x + w+offset]
                            
                            # Add hand detection to the white background image
                            imgWhiteWithHands = imgWhite.copy()
                            # Draw the hand detection on the white background
                            cv2.rectangle(imgWhiteWithHands, (rectX1, rectY1), (rectX2, rectY2), (0, 255, 0), 2)
                            
                            cv2.imshow("Hand Region", imgCropWithHands)
                            cv2.imshow("Processed Image", imgWhiteWithHands)
                except Exception as e:
                    print(f"Error processing hand image: {e}")
                
        cv2.imshow("Camera Feed", imgMain)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
