import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Create the directory if it doesn't exist
folder = "Data/N"
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Created directory: {folder}")

cap = cv2.VideoCapture(0) #zero is id no. for our web cam
offset = 20
imgSize = 300
detector = HandDetector(maxHands=1) #1 for just single hand

counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam")
        continue
        
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255 #255 is white colour no.     
        # Make sure the coordinates are within the image bounds
        imgHeight, imgWidth = img.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, imgWidth - x)
        h = min(h, imgHeight - y)

        if w > 0 and h > 0:
            try:
                # Check if the crop coordinates are valid
                if y-offset >= 0 and x-offset >= 0 and y+h+offset <= imgHeight and x+w+offset <= imgWidth:
                    imgCrop = img[y-offset:y + h+offset, x-offset:x + w+offset]
                    
                    # Check if imgCrop is not empty
                    if imgCrop.size > 0:
                        imgCropShape = imgCrop.shape
                        
                        aspectRatio = h/w
                        if aspectRatio >1:
                            k = imgSize/h
                            wCal = math.ceil(k*w)
                            imgResize = cv2.resize(imgCrop,(wCal, imgSize))
                            imgResizeShape = imgResize.shape
                            wGap = math.ceil((imgSize-wCal)/2)
                            imgWhite[:, wGap:wCal+wGap] = imgResize
                        
                        else:
                            k = imgSize/w
                            hCal = math.ceil(k*h)
                            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
                            imgResizeShape = imgResize.shape
                            hGap = math.ceil((imgSize-hCal)/2)
                            imgWhite[hGap:hCal+hGap, :] = imgResize

                        cv2.imshow("ImageCrop", imgCrop)#output for hand crop
                        cv2.imshow("ImageWhite", imgWhite)#output for hand background
            except Exception as e:
                print(f"Error processing hand image: {e}")
            
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'):
        try:
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(f"Saved image {counter}")
        except Exception as e:
            print(f"Error saving image: {e}")

cap.release()
cv2.destroyAllWindows()
