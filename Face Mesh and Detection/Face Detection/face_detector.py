import cv2 as cv 
import mediapipe as mp
import time 

cap = cv.VideoCapture("Face Detection\\Videos\\roller_coaster (720p).mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.85)     # min_detection_confidence =0.85 so no false faces detection

while True:
    isTrue ,frame = cap.read()

    imgRGB = cv.cvtColor(frame , cv.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results.detection)
    if results.detections:
        for id , detection in enumerate(results.detections):     # result.detection --> detectsfaces, id - Face number
            # mpDraw.draw_detection(frame, detection)
            # print(id,detection)                                # for face label_id = 0  # score - confidence it is face 
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)    # xmin , ymin , h , w                                                    
            # bounding box from class detection
            bboxC = detection.location_data.relative_bounding_box
            # OWN bounding box
            ih , iw , ic = frame.shape
            bbox = int(bboxC.xmin * iw) , int(bboxC.ymin*ih) ,\
                    int(bboxC.width * iw) , int(bboxC.height*ih) 
            cv.rectangle(frame ,bbox, (255,0,255),thickness=2)
            cv.putText(frame, f"{int(detection.score[0]*100)} %",(bbox[0],bbox[1]-20) , cv.LINE_AA , 2, (255,0,255),thickness=2) 



    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv.putText(frame, f"FPS {int(fps)}",(20,70) , cv.FONT_HERSHEY_COMPLEX , 3, (255,0,0) ) 

    cv.imshow("Frame",frame)

    if cv.waitKey(5)  & 0xFF == ord("x"):
        break
