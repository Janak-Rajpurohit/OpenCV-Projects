import os 
import time
import cv2 as cv
import HandTrackingModule as htm


wCam , hCam = 640,480
cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

# 6.jpeg = 0
folder_path = "Finger Counting\\finger photos"
myList = os.listdir(folder_path)
print(myList)
overlayList = []
# converting img into array and append to list 
for img in myList:
    image = cv.imread(os.path.join(folder_path,img))
    image = cv.resize(image,(200,200),interpolation = cv.INTER_AREA)
    overlayList.append(image)
print(len(overlayList))


pTime = 0
detector = htm.HandDetector(detectionCon=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:
    isTrue , frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame,draw=False)
    # print(lmList)

    if len(lmList)!=0:
            # tip of fingers - 4 thumb , 8 index , 12 middle , 16 ring , 20 pinky
            #  checking is they are below particular landmark then counting
            fingers = []
            # for thumb 
            if lmList[tip_ids[0]][1] > lmList[tip_ids[0]-1][1]:          # eg if above point is on left the below point thumb is close
                # print("Thumb is open")
                fingers.append(1)
            else:
                fingers.append(0)
            # For Other fingers
            for id in range(1, len(tip_ids)):    #ignoring thumb 
                if lmList[tip_ids[id]][2] < lmList[tip_ids[id]-2][2]:          # eg 8 < 8-2  then open   , for thumb we consisder x - axis
                    # print("Index finger is open")
                    fingers.append(1)
                else:
                     fingers.append(0)
            
            # print(fingers)
            total_fingers = fingers.count(1)       # count the no of ones 
            print(total_fingers)
            h,w,c = overlayList[total_fingers-1].shape
            frame[0:h , 0:w] = overlayList[total_fingers-1]          # when total_finger = 0 then it give -1 i.e..  last element

            cv.rectangle(frame , (20,225), (170,400) , (0,255,0),cv.FILLED)
            cv.putText(frame,f"{total_fingers}",(45,375),cv.FONT_HERSHEY_PLAIN,10,(255,0,0),10)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(frame,f"FPS: {int(fps)}",(500,60),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
    cv.imshow("Finger Counter",frame)

    if cv.waitKey(3) & 0xFF==ord("x"):
        break