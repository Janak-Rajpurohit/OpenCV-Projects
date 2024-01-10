import cv2 as cv 
import mediapipe as mp
import time

class FaceMeshDetector:
    def __init__(self,staticMode=False , maxFaces=2,minDetectionCon=0.5,minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon


        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.FaceMesh = self.mpFaceMesh.FaceMesh(static_image_mode = self.staticMode ,max_num_faces = self.maxFaces \
                                                 ,min_detection_confidence = self.minDetectionCon  ,min_tracking_confidence = self.minTrackCon )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)       # changing thickness of mesh , total 468 points


    def Mesh(self,img,draw=True):
        self.imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.FaceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLMS in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,faceLMS,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)       #FACEMESH_CONTOURS - for points only
                
                face = []
                for id ,lm in enumerate(faceLMS.landmark):
                    # print(lm)
                    ih , iw , ic = img.shape
                    x,y = int(lm.x * iw)  , int(lm.y * ih)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)
        return img,faces                        # faces - list of faces landmarks list


def main():
    cap = cv.VideoCapture("Face Detection\\Videos\\roller_coaster (720p).mp4")
    pTime = 0

    fm = FaceMeshDetector()

    while True:
        isTrue,frame = cap.read()
        frame ,faces = fm.Mesh(frame)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv.putText(frame,f"{int(fps)}",(20,70),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv.imshow("Frame",frame)
        

        if cv.waitKey(7) & 0xFF==ord("x"):
            break




if __name__ == "__main__":
    main()