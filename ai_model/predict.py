import cv2
import os
import pickle
import numpy as np
import mediapipe as mp

class Predict:
    def __init__(self):
        self.ROOT_DIR = "C:\\Users\\murat\\Documents\\tensorflow\\body_language_decoder"
        self.model = f"{self.ROOT_DIR}\\model\\body_language.pkl"
        
        self.drawing = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.face_mesh = self.mpFaceMesh.FaceMesh()
    
    def predict(self):
        video = cv2.VideoCapture(0)
        
        with self.mpFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while video.isOpened():
                retval, frame = video.read()
                frame = cv2.flip(src=frame, flipCode=1)
                results = face_mesh.process(frame)
                
                if results.multi_face_landmarks:
                    
                    for landmarks in results.multi_face_landmarks:
                        landmarkList = []
                        for id, landmark in enumerate(landmarks.landmark):
                            height, width, color = frame.shape
                            cx, cy = int(landmark.x * width), int(landmark.y * height)
                            landmarkList.extend([cx, cy])
                    
                    with open(self.model, "rb") as f:
                        model = pickle.load(f)
                    
                    y_pred = model.predict(np.array([landmarkList]))
                    y_prob = model.predict_proba(np.array([landmarkList]))[0]
                    #print(y_pred, y_prob[0])
                    
                    # if y_prob[0][0] > y_prob[0][1]:
                    #     cv2.putText(img=frame, text="Gulmuyorsun", org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
                    # else:
                    #     cv2.putText(img=frame, text="Guluyorsun", org=(20, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1)
                    cv2.putText(img=frame, text=f"{y_pred[0]}, {max(y_prob)}", org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                    print(y_pred)
                cv2.imshow(winname="Predict", mat=frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    video.release()
                    cv2.destroyAllWindows()
    
p1 = Predict()
p1.predict()