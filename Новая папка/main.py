import cv2
import dlib
import numpy as np
import face_recognition
from PyQt5 import QtCore
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

        # Инициализация модели распознавания лиц
        shape_predictor_path = "face_recognition_models/shape_predictor_68_face_landmarks.dat"
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.recognizer = dlib.face_recognition_model_v1("face_recognition_models/dlib_face_recognition_resnet_model_v1.dat")

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

    def load_images(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, "Выберите фотографии", "", "Images (*.png *.xpm *.jpg *.bmp)")
        for filename in filenames:
            image = face_recognition.load_image_file(filename)
            face_locations = face_recognition.face_locations(image)
            for location in face_locations:
                top, right, bottom, left = location
                face_shape = dlib.rectangle(left, top, right, bottom)
                face_descriptor = np.array(face_recognition.face_encodings(image, [location])[0])
                self.sample_face_descriptors.append(face_descriptor)

    def start_video(self):
        self.load_images()
        self.timer.start(20)

    def stop_video(self):
        self.timer.stop()

    def process_frame(self, frame, faces, recognizer, threshold):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faces(gray_frame, 0)
        face_locations = []
        for detected_face in detected_faces:
            face_shape = self.predictor(gray_frame, detected_face).parts()
            face_locations.append(detected_face)
            x, y, w, h = detected_face.left(), detected_face.top(), detected_face.width(), detected_face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Сравнение дескрипторов лиц на видеопотоке с образцами лиц
            face_descriptor = np.array(face_recognition.face_encodings(gray_frame, [detected_face])[0])
            distances = face_recognition.face_distance(face_descriptor, self.sample_face_descriptors)
            min_distance = min(distances)
            if min_distance < threshold:
                index = np.argmin(distances)
                name = "Sample face {}".format(index + 1)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return face_locations

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            # Обработка кадра
            face_locations = self.process_frame(frame, self.faces, self.recognizer, 0.6)
            # Отображение кадра на GUI
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
            self.label.setGeometry(0, 0, width, height)
            self.label.setAlignment(QtCore.Qt.AlignCenter)


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()