import mediapipe as mp 
import cv2 as cv
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()    #only uses rgb imgs
mpDraw =  mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

while True:
    isTrue , frame = cap.read()
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # print(id, lm)    # it give ratio to img of landmarks so we multiply with width and height  
                h , w, c= frame.shape
                # coordinates of landmark
                cx , cy =int(lm.x*w) , int(lm.y*h)
                print(id,cx,cy)
                if id == 0 :
                    cv.circle(frame , (cx,cy), 15 ,(255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(frame,handLms, mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime
    cv.putText(frame, str(int(fps)),(10,70), cv.FONT_HERSHEY_PLAIN, 3, (255,0,255))

    cv.imshow("Frames",frame)
    if cv.waitKey(1) & 0xFF==ord("d"):
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(0)



