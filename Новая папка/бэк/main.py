import cv2
import dlib
import numpy as np
import face_recognition
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = "Face Recognition App"
        self.initUI()

        # Список дескрипторов образцов лиц
        self.sample_face_descriptors = []

    def initUI(self):
        self.setWindowTitle(self.title)

        # Виджет для отображения видеопотока
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 640, 480)

        # Кнопки управления
        self.load_button = QPushButton("Загрузить фотографии", self)
        self.load_button.setGeometry(0, 480, 200, 40)
        self.load_button.clicked.connect(self.load_images)

        self.start_button = QPushButton("Старт", self)
        self.start_button.setGeometry(200, 480, 200, 40)
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("Стоп", self)
        self.stop_button.setGeometry(400, 480, 200, 40)
        self.stop_button.clicked.connect(self.stop_video)

        # Таймер для обновления видеопотока
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video)

        # Веб-камера и обработка лиц
        self.cap = cv2.VideoCapture(0)
        self.faces = dlib.get_frontal_face_detector()
        self.recognizer = dlib.face_recognition_model_v1("face_recognition_models/face_recognition_model_v1.dat")

    def load_images(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Выберите фотографии", "", "Images (*.png *.xpm *.jpg *.bmp)")
        for filename in filenames:
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            face_descriptors = face_recognition.face_encodings(image, face_locations)
            self.sample_face_descriptors.extend(face_descriptors)

    def start_video(self):
        self.timer.start(20)
        self.load_images()

    def stop_video(self):
        self.timer.stop()

    def process_frame(self, frame, faces, recognizer, threshold):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faces(gray_frame)
        for face in detected_faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_descriptor = recognizer.compute_face_descriptor(gray_frame, face)
            # Сравнение дескрипторов лиц на видеопотоке с образцами лиц
            distances = face_recognition.face_distance(self.sample_face_descriptors, face_descriptor)
            min_distance = min(distances)
            if min_distance < threshold:
                index = np.argmin(distances)
                name = "Sample face {}".format(index + 1)
                cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def update_video(self):
        _, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.process_frame(frame, self.faces, self.recognizer, threshold=0.6) # Используйте подходящее пороговое значение
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qimage = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()