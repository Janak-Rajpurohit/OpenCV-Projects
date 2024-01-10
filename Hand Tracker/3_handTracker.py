import mediapipe as mp 
import cv2 as cv
import time

import handTrackingModule as htm

prevTime = 0
currentTime = 0
cap = cv.VideoCapture(0)

detector = htm.HandDetector()
while True:
    isTrue , frame = cap.read()

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    if len(lmList)!=0:
        print(lmList[4])              #lmList[index_no]  // thumb
    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime
    cv.putText(frame, str(int(fps)),(10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255))

    cv.imshow("Frames",frame)
    if cv.waitKey(1) & 0xFF==ord("d"):
        break

cv.destroyAllWindows()
cv.waitKey(0)