import torch

import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections
from supervision import BoxAnnotator

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib


class ObjectDetection:

    

    def __init__(self):

        self.capture_index = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(color=ColorPalette.DEFAULT, thickness=2)

    def load_model(self):

        model = YOLO("best.pt")
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # print("empty confidence=",confidences)

        # Extract detections for person class
        for result in results[0]:
            # print("data confidence=", confidences)
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if result == 0:
                print("smoking not detected")

            else:
                print("smoking detected")
                cv2.imwrite('smoke.jpg', frame)
                print("email sent along with image capture........")

                # self.smoke_image_mail_sender()

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy().astype(int))
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
                # print("x confidence=",confidences)

        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )


        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections)

        return frame

    def __call__(self):

        cap = cv2.VideoCapture(0)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            cv2.imshow('Smoking Detection', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection()
detector()
