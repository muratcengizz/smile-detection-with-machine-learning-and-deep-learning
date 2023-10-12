import mediapipe as mp
import cv2
import csv
import os
import numpy as np

class CaptureLandmarks:
    def __init__(self):
        self.num_coords = 468
        self.ROOT_DIR = "C:\\Users\\murat\\Documents\\tensorflow\\body_language_decoder"
        self.csv_file = os.path.join(self.ROOT_DIR, "dataset")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.class_name = "smile"
        
    def create_csv_table(self):
        landmarks = ["class"]
        for value in range(1, self.num_coords+1):
            landmarks += [f'x{value}', f"y{value}"]
        
        with open(f"{self.csv_file}\\coords.csv", mode="w", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)
        
    
    def capture_landmarks_and_export_csv(self):
        video = cv2.VideoCapture(0)
        
        with self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            while video.isOpened():
                retval, frame = video.read()
                frame = cv2.flip(src=frame, flipCode=1)
                image = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                results = face_mesh.process(image)
                
                if results.multi_face_landmarks:
                    image.flags.writeable = True
                    image = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2BGR)
                    
                    for landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(image=image, landmark_list=landmarks, connections=self.mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1))
                        face_row = [self.class_name]
                        for id, landmark in enumerate(landmarks.landmark):
                            height, width, color = image.shape
                            cx, cy = int(landmark.x * width), int(landmark.y * height)
                            face_row.extend([cx, cy])
                        
                        
                        try:
                            
                            with open(f"{self.csv_file}\\coords.csv", mode="a", newline="") as f:
                                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                csv_writer.writerow(face_row)
                                    
                        except Exception as e:
                            print("hata", e)
                # face landmarks
                
                
                
                try:
                   pass
                except:
                    pass
                
                cv2.imshow(winname="Capture Landmarks", mat=image)
                
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    video.release()
                    cv2.destroyAllWindows()

p1 = CaptureLandmarks()
p1.capture_landmarks_and_export_csv()