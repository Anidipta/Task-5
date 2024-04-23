import cv2
import numpy as np
from keras.preprocessing import image
import warnings
from keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

# Load the model
model = load_model("ml models/model_5.h5")

# Load the cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        self.emotion_label = tk.Label(window, text="")
        self.emotion_label.pack(anchor=tk.CENTER, expand=True)

        self.delay = 10
        self.update()

        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + h, x:x + w]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (224, 224))  # Resize to match model input shape
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                # find max indexed array                
                max_index = np.argmax(predictions[0])

                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                predicted_emotion = emotions[max_index]

                self.emotion_label.config(text=predicted_emotion)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("snapshot.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

App(tk.Tk(), "Facial emotion analysis")
