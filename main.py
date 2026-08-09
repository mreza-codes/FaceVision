import insightface
from insightface.app import FaceAnalysis
import os
import cv2
import numpy as np
from ui_facevision import Ui_main_window
from PyQt6 import QtWidgets,QtCore,QtGui
import pickle
import onnxruntime as ort

base_path = os.path.dirname(os.path.abspath(__file__))


class MainApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_main_window()
        self.ui.setupUi(self)
        providers = ["CPUExecutionProvider"]
        ctx_id = -1

        try:
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider"]
                ctx_id = 0
                print("CUDA detected → using GPU")
            else:
                print("CUDA not detected → using CPU")
        except:
            print("ONNXRuntime not available → using CPU")

        # InsightFace CPU/GPU auto mode
        self.face_app = FaceAnalysis(name="buffalo_l" , providers =providers, download = True , root =os.path.join(base_path, "models"))
        self.face_app.prepare(ctx_id=ctx_id)

        #database:
        self.database = {}
        if os.path.exists("faces_db.pkl"):
            with open("faces_db.pkl","rb") as f:
                self.database = pickle.load(f)
        else:
            self.database = {}

            #saving the first database:
            with open("faces_db.pkl","wb") as f:
                pickle.dump(self.database,f)



        self.cap = None  #openning webcam

        #creating timer:
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)

        #connceting the buttons:
        self.ui.btn_add_person.clicked.connect(self.add_person)
        self.ui.btn_start.clicked.connect(self.start_camera)
        self.ui.btn_stop.clicked.connect(self.stop_camera)


    def start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #openning webcam
        # if the cloent doesn't have webcam:
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.warning(self, "Error", "No webcam detected!")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.timer.start(30)
        
    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        

    def update_frame(self):
        if self.cap is None:
            return
        ret , frame = self.cap.read()
        if not ret:
            return
        
        # face Recognition:
        faces = self.face_app.get(frame)
        for face in faces:
            x1,y1,x2,y2 = face.bbox.astype(int)
            emb = face.embedding
            best_name = "Unknown"
            best_score = -1.0

            # Comparing to database:
            for name , db_emb in self.database.items():
                score = self.cosine_similarity(emb , db_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
            if best_score < 0.35:
                best_name = "Unknown"

            # creating rectangle:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #writing names:
            cv2.putText(frame, best_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        #changing frame to QImage for showing in QLabel:
        frame = cv2.resize(frame, (self.ui.camera_label.width(), self.ui.camera_label.height()))
        self.last_frame = frame.copy()   #saving the last frame for add_person
        rgb_image = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(qt_image)
        
        #show in QLabel
        self.ui.camera_label.setPixmap(pixmap)


    def add_person(self):

        # if we dont have last frame(client didn't click the start camera btn) do this:
        if not hasattr(self,"last_frame"):
            QtWidgets.QMessageBox.warning(self , "Warning", "Camera is not started yet!")
            return
        
        name , ok = QtWidgets.QInputDialog.getText(self,"Add Person","Enter Name:")
        if not ok or not name.strip():
            return  #if client didn't write anything or canceled

        # getting last frame from webcam:
        frame = self.last_frame.copy()  

        #changing frame to RGB:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_app.get(rgb_image) #finding the face from image
        if len(faces) == 0:
            QtWidgets.QMessageBox.warning(self,"Warning","No face detected!")
            return
        face = faces[0]  #gets the first face in picture
        embedding = face.embedding  #gets the embedding of face
        
        self.database[name] = embedding  #saving name in database
        with open("faces_db.pkl","wb") as f:
            pickle.dump(self.database,f)
        QtWidgets.QMessageBox.information(self,"Saved",f"{name} added successfully!")


   

    #این تابع دو بردار رو مقایسه میکنه  و عددی بین 0 تا 1 میدهد و هرچه شباهت بیشتر باشد عدد هم بزرگتر 
    def cosine_similarity(self,a,b):
        return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))  



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainApp()
    window.show()
    app.exec()