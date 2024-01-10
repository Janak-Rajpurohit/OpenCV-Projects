import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import handTrackingModule  as htm
import math
# for volume control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam ,hCam = 640 , 480

cap = cv.VideoCapture(0)
cap.set(3,wCam)       #setting width and height
cap.set(4,hCam)
pTime=0

detector = htm.HandDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
vol_range = volume.GetVolumeRange()        # range is -65.25 to 0   i.e.. for 0 to 100 vol
# volume.SetMasterVolumeLevel(0, None)
minVol = vol_range[0]
maxVol = vol_range[1]



while True:
    isTrue , frame = cap.read()
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame,draw=False)
    if len(lmlist)!=0:
        # print(lmlist[4],lmlist[8])
        # [2 , 200, 300]
        x1,y1 = lmlist[4][1] , lmlist[4][2]
        x2,y2 = lmlist[8][1] , lmlist[8][2]
        cx , cy = (x1+x2)//2 , (y1+y2)//2                                                 # getting center of line
        cv.circle(frame,(x1,y1),10,(255,0,255),cv.FILLED)
        cv.circle(frame,(x2,y2),10,(255,0,255),cv.FILLED)
        cv.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
        cv.circle(frame,(cx,cy),7,(255,0,255),cv.FILLED)
        length = math.hypot(x2-x1 , y2-y1)                                                # length of line
        # print(length)
        # max length - 250 ,  min length - 30
        #  vol range - -65 , 0
        vol = np.interp(length,[30,250],[minVol,maxVol])
        print(int(length),vol) 
        volume.SetMasterVolumeLevel(vol, None)

        if length<40:                 # center point turns green when length of line < 30
                cv.circle(frame,(cx,cy),7,(0,255,0),cv.FILLED)
            
        vol_bar = np.interp(length,[30,250],[400,150])
        vol_per = np.interp(length,[30,250],[0,100])
        cv.rectangle(frame,(50,150),(85,400),(255,0,0),3)
        cv.rectangle(frame,(50,int(vol_bar)),(85,400),(255,0,0),cv.FILLED)
        cv.putText(frame,f"Vol: {int(vol_per)}",(40,450),cv.FONT_HERSHEY_PLAIN,2,(255,0,0),2)




    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime=cTime 

    cv.putText(frame,f"FPS : {int(fps)}",(35,60),cv.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv.imshow("Frame",frame)

    if cv.waitKey(3) & 0xFF==ord("x"):
        break
