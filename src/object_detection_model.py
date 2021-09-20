import logging
import time

import requests

from src.constants import classes, landing_statuses
from src.detected_object import DetectedObject

import cv2
import numpy as np


class ObjectDetectionModel:
    # Base class for team models

    def __init__(self, evaluation_server_url):
        logging.info('Created Object Detection Model')
        self.evaulation_server = evaluation_server_url

        self.uap_uai_cfg = "cfg/uap_uai.cfg"
        self.uap_uai_weights = "weights/uap_uai_3.weights"

        self.insan_arac_cfg = "cfg/insan_arac.cfg"
        self.insan_arac_weights = "weights/insan_arac_2.weights"

        self.uap_uai_classes = "data/uap_uai.names"
        self.insan_arac_classes = "data/insan_arac.names"

        self.uap_uai_image_dim = 256
        self.insan_arac_image_dim = 416

        self.confThreshold = 0.5
        self.nmsThreshold = 0.5

        with open(self.uap_uai_classes, 'rt') as f:
            self.uap_uai_class_names = f.read().rstrip('\n').split('\n')

        with open(self.insan_arac_classes, 'rt') as f:
            self.insan_arac_class_names = f.read().rstrip('\n').split('\n')

        self.uap_uai_net = cv2.dnn.readNetFromDarknet(self.uap_uai_cfg, self.uap_uai_weights)
        self.insan_arac_net = cv2.dnn.readNetFromDarknet(self.insan_arac_cfg, self.insan_arac_weights)

        self.uap_uai_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.uap_uai_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.insan_arac_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.insan_arac_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @staticmethod
    def download_image(img_url, images_folder):
        t1 = time.perf_counter()
        img_bytes = requests.get(img_url).content
        image_name = img_url.split("/")[-1]  # frame_x.jpg

        with open(images_folder + image_name, 'wb') as img_file:
            img_file.write(img_bytes)

        t2 = time.perf_counter()

        logging.info(f'{img_url} - Download Finished in {t2 - t1} seconds to {images_folder + image_name}')

    def process(self, prediction):
        # Yarışmacılar resim indirme, pre ve post process vb işlemlerini burada gerçekleştirebilir.
        # Download image (Example)
        self.download_image(prediction.image_url, "./_images/")
        # Örnek: Burada OpenCV gibi bir tool ile preprocessing işlemi yapılabilir. (Tercihe Bağlı)
        # ...
        # Nesne tespiti modelinin bulunduğu fonksiyonun (self.detect() ) çağırılması burada olmalıdır.
        frame_results = self.detect(prediction)
        # Tahminler objesi FramePrediction sınıfında return olarak dönülmelidir.
        return frame_results

    def detect(self, prediction):
        # Modelinizle bu fonksiyon içerisinde tahmin yapınız.
        # results = self.model.evaluate(...) # Örnektir.

        image_name = prediction.image_url.split("/")[-1]  # frame_x.jpg
        image = cv2.imread("_images/" + image_name)

        uap_uai_blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.uap_uai_image_dim, self.uap_uai_image_dim),
                                             (0, 0, 0), swapRB=True, crop=False)
        insan_arac_blob = cv2.dnn.blobFromImage(image, 1 / 255, (self.insan_arac_image_dim, self.insan_arac_image_dim),
                                                (0, 0, 0), swapRB=True, crop=False)

        self.uap_uai_net.setInput(uap_uai_blob)
        self.insan_arac_net.setInput(insan_arac_blob)

        layer_names_1 = self.uap_uai_net.getLayerNames()
        output_names_1 = [layer_names_1[i[0] - 1] for i in self.uap_uai_net.getUnconnectedOutLayers()]

        layer_names_2 = self.insan_arac_net.getLayerNames()
        output_names_2 = [layer_names_2[i[0] - 1] for i in self.insan_arac_net.getUnconnectedOutLayers()]

        uap_uai_outputs = self.uap_uai_net.forward(output_names_1)
        insan_arac_outputs = self.insan_arac_net.forward(output_names_2)

        uap_uai_indices, uap_uai_bbox, uap_uai_class_ids, uap_uai_confs = self.find_objects(uap_uai_outputs, image,
                                                                                           self.uap_uai_class_names)
        insan_arac_indices, insan_arac_bbox, insan_arac_class_ids, insan_arac_confs = self.find_objects(
            insan_arac_outputs, image, self.insan_arac_class_names)

        # iterate in uap_uai results
        for i in uap_uai_indices:
            i = i[0]
            box = uap_uai_bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imwrite("box_images/" + image_name, image)

            top_left_x, top_left_y, bottom_right_x, bottom_right_y = x, y, x + w, y + h
            class_id = uap_uai_class_ids[i]
            if class_id == 0 or class_id == 1:
                cls = classes["UAP"]
            else:
                cls = classes["UAI"]
            if class_id == 0 or class_id == 2:
                landing_status = landing_statuses["Inilebilir"]
            else:
                landing_status = landing_statuses["Inilemez"]
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        # iterate in insan_arac results
        for i in insan_arac_indices:
            i = i[0]
            box = insan_arac_bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.imwrite("box_images/" + image_name, image)

            top_left_x, top_left_y, bottom_right_x, bottom_right_y = x, y, x + w, y + h
            class_id = insan_arac_class_ids[i]
            if class_id == 0:
                cls = classes["Insan"]
            else:
                cls = classes["Tasit"]
            landing_status = landing_statuses["Inis Alani Degil"]
            d_obj = DetectedObject(cls,
                                   landing_status,
                                   top_left_x,
                                   top_left_y,
                                   bottom_right_x,
                                   bottom_right_y)
            prediction.add_detected_object(d_obj)

        return prediction

    def find_objects(self, outputs, img, class_names):
        h_t, w_t, c_t = img.shape
        bbox = []
        class_ids = []
        confs = []

        for output in outputs:
            for det in output:
                scores = det[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confThreshold:
                    w, h = int(det[2] * w_t), int(det[3] * h_t)
                    x, y = int(det[0] * w_t - w / 2), int(det[1] * h_t - h / 2)
                    bbox.append([x, y, w, h])
                    class_ids.append(class_id)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nmsThreshold)

        return indices, bbox, class_ids, confs
