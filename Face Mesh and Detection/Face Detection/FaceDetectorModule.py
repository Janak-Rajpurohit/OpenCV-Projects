import cv2 as cv 
import mediapipe as mp
import time 


class FaceDetector:

    def __init__(self,min_detection_confidence =0.50):
        self.min_detection_confidence = min_detection_confidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)     # min_detection_confidence =0.85 so no false faces detection

    def faceDetect(self, img , draw=True):
        bbox_corr = []
        imgRGB = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)
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
                ih , iw , ic = img.shape
                bbox = int(bboxC.xmin * iw) , int(bboxC.ymin*ih) ,\
                        int(bboxC.width * iw) , int(bboxC.height*ih) 
                bbox_corr.append(bbox)

                if draw:
                    cv.rectangle(img ,bbox, (255,0,255),thickness=2)
                    cv.putText(img, f"{int(detection.score[0]*100)} %",(bbox[0],bbox[1]-20) , cv.LINE_AA , 2, (255,0,255),thickness=2) 
        return img , bbox_corr

def main():
    cap = cv.VideoCapture("Face Detection\\Videos\\roller_coaster (720p).mp4")
    pTime = 0
    fd = FaceDetector()
    while True:
        isTrue ,frame = cap.read()

        frame , bbox_corr= fd.faceDetect(frame)
        # print(bbox_corr)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(frame, f"FPS {int(fps)}",(20,70) , cv.FONT_HERSHEY_COMPLEX , 3, (255,0,0) ) 

        cv.imshow("Frame",frame)

        if cv.waitKey(5)  & 0xFF == ord("x"):
            break


if __name__ == '__main__':
    main()
