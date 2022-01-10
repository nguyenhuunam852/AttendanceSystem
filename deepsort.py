from scipy.stats import multivariate_normal
from tensorflow.python.util.tf_export import api_export
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.application_util import preprocessing as prep
from deep_sort.deep_sort.detection import Detection
import tensorflow as tf
import numpy as np
import cv2


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def _run_in_batches(f, data_dict, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
        out[s:e] = f(batch_data_dict)
    if e < len(out):
        batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
        out[e:] = f(batch_data_dict)


class ImageEncoder(object):
    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="Identity"):
        self.session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))

        with tf.compat.v1.gfile.GFile(checkpoint_filename, "rb") as file_handle:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(file_handle.read())

        with tf.device('/GPU:0'):
            tf.import_graph_def(graph_def, name="net")
            self.input_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "%s:0" % input_name)
            self.output_var = tf.compat.v1.get_default_graph().get_tensor_by_name(
                "%s:0" % output_name)

        assert len(self.output_var.get_shape()) == 2
        assert len(self.input_var.get_shape()) == 4
        self.feature_dim = self.output_var.get_shape().as_list()[-1]
        self.image_shape = self.input_var.get_shape().as_list()[1:]

    def __call__(self, data_x, batch_size=32):
        out = np.zeros((len(data_x), self.feature_dim), np.float32)
        _run_in_batches(
            lambda x: self.session.run(self.output_var, feed_dict=x),
            {self.input_var: data_x}, out, batch_size)
        return out


def extract_image_patch(image, bbox, patch_shape):
    bbox = np.array(bbox)
    if patch_shape is not None:
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sx:ex, sy:ey]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    # cv2.imwrite('test.png', image)
    image = image.astype(np.float32) / 255.
    # cv2.imwrite('test.png', image)
    return image


class deepsort_rbc:
    def __init__(self, model):
        self.encoder = self.create_box_encoder(model, batch_size=32)
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", .5, 100)
        self.tracker = Tracker(self.metric)

    def create_box_encoder(self, model_filename, input_name="images", output_name="Identity", batch_size=32):
        image_encoder = ImageEncoder(model_filename, input_name, output_name)
        image_shape = image_encoder.image_shape

        def encoder(image, boxes):
            image_patches = []
            for i, box in enumerate(boxes):
                patch = extract_image_patch(
                    image, box, image_shape[:2])
                if patch is None:
                    print("WARNING: Failed to extract image patch: %s." %
                          str(box))
                    patch = np.random.uniform(
                        0., 255., image_shape).astype(np.uint8)
                image_patches.append(patch)

            image_patches = np.asarray(image_patches)
            return image_encoder(image_patches, batch_size)

        return encoder

    def run_deep_sort(self, frame, out_scores, out_boxes):
        if out_boxes == []:
            self.tracker.predict()
            print('No detections')
            trackers = self.tracker.tracks
            return trackers

        detections = np.array(out_boxes)

        with tf.device('/GPU:0'):
            features = l2_norm(self.encoder(frame, detections))

        dets = [Detection(bbox, score, feature) for bbox, score,
                feature in zip(detections, out_scores, features)]

        outboxes = np.array([d.tlwh for d in dets])
        outscores = np.array([d.confidence for d in dets])

        indices = prep.non_max_suppression(outboxes, 0.8, outscores)

        dets = [dets[i] for i in indices]
        self.tracker.predict()
        self.tracker.update(dets)
        return self.tracker, dets
