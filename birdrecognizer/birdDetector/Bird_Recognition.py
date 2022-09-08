import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2


class birdRecognizer():
    def __init__(self, classifier_path, object_detector_path, class_names_path):
        # load tensorflow models from paths
        self.classifier = load_model(classifier_path)
        self.object_detector = load_model(object_detector_path)

        # get class names from .txt file
        self.class_names = []
        with open(class_names_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.class_names.append(line.strip()[1:-2])

    def extract_bird_roi(self, img, threshold=0.5):
        # get image dimensions
        width, height = img.shape[0], img.shape[1]

        # perform object detection
        detector_output = self.object_detector(np.expand_dims(img, axis=0))

        # bird class is 16th in the coco dataset
        bird_mask = detector_output["detection_classes"] == 16
        # apply threshold
        bird_mask = (
            detector_output["detection_scores"] > threshold) & bird_mask
        bird_boxes = detector_output["detection_boxes"][bird_mask]

        bird_images = []
        # this will be a list of tuples with pair:
        # ROI (Region Of Interest), Box coordinates

        # extract ROI and coordinates for each detected bird
        for box in bird_boxes:
            min_corner = (int(box[1] * height), int(box[0] * width))
            max_corner = (int(box[3] * height), int(box[2] * width))
            bird_images.append((img[min_corner[1]: max_corner[1], min_corner[0]: max_corner[0]],
                                (min_corner, max_corner)))

        return bird_images

    def identify_species(self, rois):

        predictions = []  # list of bird species identified

        for roi in rois:
            roi_resized = tf.image.resize(roi[0] / 255.0, (224, 224))
            prediction = tf.argmax(self.classifier.predict(
                tf.expand_dims(roi_resized, axis=0)), axis=1)
            predictions.append((self.class_names[int(prediction)]))

        return predictions

    def draw_bounding_boxes_with_text(self, img, rois, predictions):
        image_copy = img.copy()

        for n, roi in enumerate(rois):
            cv2.rectangle(image_copy, roi[1][0], roi[1][1], (0, 255, 0))
            cv2.putText(image_copy, predictions[n], (roi[1][0][0], roi[1][0][1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return image_copy

    def detect_bird_species(self, image):
        # detect birds in the image
        rois = self.extract_bird_roi(image)

        # identify bird species
        species = self.identify_species(rois)

        # draw bounding boxes and text
        output_image = self.draw_bounding_boxes_with_text(image, rois, species)

        return output_image


# recognizer = birdRecognizer(classifier_path="ML_modeling/400_bird_species_EFFNetB0",
#                             object_detector_path="ML_modeling/Efficientdet_d2",
#                             class_names_path="ML_modeling\class_names.txt")
