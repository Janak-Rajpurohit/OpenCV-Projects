import mediapipe as mp 
import cv2 as cv
import time

class HandDetector():
    def __init__(self , mode=False , maxHands=2,modelComplex=1 , detectionCon=0.5 , trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)       #only uses rgb imgs
        self.mpDraw =  mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:        #for all hands
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms, self.mpHands.HAND_CONNECTIONS)  
        return img  
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]     #particular hand
            for id,lm in enumerate(myhand.landmark):
                # print(id, lm)    # it give ratio to img of landmarks so we multiply with width and height  
                h , w, c= img.shape
                # coordinates of landmark                
                cx , cy =int(lm.x*w) , int(lm.y*h)
                # print(id,cx,cy)
                lmList.append([id,cx,cy])
                if draw :
                    cv.circle(img , (cx,cy), 7 ,(255,0,255), cv.FILLED)
        return lmList

def main():
    prevTime = 0
    currentTime = 0
    cap = cv.VideoCapture(0)

    detector = HandDetector()
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


#  NOW WE CAN IMPORT THIS FILE INTO OTHER AND USE CLASS HandDetector with
# handTrackingModule.HandDetector()
if __name__ == '__main__':
    main()



