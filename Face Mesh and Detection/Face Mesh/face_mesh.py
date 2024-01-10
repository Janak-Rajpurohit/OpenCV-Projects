import cv2 as cv 
import mediapipe as mp
import time

cap = cv.VideoCapture("Face Detection\\Videos\\roller_coaster (720p).mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)       # changing thickness of mesh , total 468 points

while True:
    isTrue,frame = cap.read()
    imgRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLMS in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,faceLMS,mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)       #FACEMESH_CONTOURS - for points only

            for id ,lm in enumerate(faceLMS.landmark):
                # print(lm)
                ih , iw , ic = frame.shape
                x,y = int(lm.x * iw)  , int(lm.y * ih)
                print(id,x,y)
                


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame,f"{int(fps)}",(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
    cv.imshow("Frame",frame)
    

    if cv.waitKey(7) & 0xFF==ord("x"):
        break