from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder
import tensorflow as tf
import numpy as np
import cv2


def check_overlap(curr, li):
    for va in li:
        # overlap
        if abs(va - curr) < 1000:
            return 1
    return 0


class Detector(object):
    def __init__(self, path_config, path_ckpt, path_to_labels):
        self.path_config = path_config
        self.path_ckpt = path_ckpt
        self.label_path = path_to_labels

        self.category_index = label_map_util.create_category_index_from_labelmap(
            path_to_labels, use_display_name=True)
        self.detection_model = self.load_model()

        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def detect_fn(self, image):
        with tf.device('/device:GPU:0'):
            image, shapes = self.detection_model.preprocess(image)
            prediction_dict = self.detection_model.predict(image, shapes)
            detections = self.detection_model.postprocess(
                prediction_dict, shapes)
        return detections

    def load_model(self):
        configs = config_util.get_configs_from_pipeline_file(self.path_config)
        model_config = configs['model']
        detection_model = model_builder.build(
            model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(self.path_ckpt).expect_partial()

        return detection_model

    def predict(self, image, height, width, threshold=0.6):
        image = np.asarray(image)
        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        detections['detection_classes'] = detections['detection_classes'].astype(
            np.int64)
        self.detection_scores = detections['detection_scores']
        self.detection_classes = detections['detection_classes']
        self.detection_boxes = detections['detection_boxes']

        detections = np.array(self.detection_boxes)
        out_scores = np.array(self.detection_scores)

        set_detection = []
        set_out_scores = []

        for index, item in enumerate(detections):
            if out_scores[index] > threshold:
                min_point = item[:2]
                min_point = min_point * (width, height)
                max_point = item[2:]
                max_point = max_point * (width, height)
                set_detection.append(
                    np.concatenate([min_point, max_point - min_point]))
                set_out_scores.append(out_scores[index])

        set_detection = np.array(set_detection)
        set_out_scores = np.array(set_out_scores)

        return set_detection, set_out_scores


