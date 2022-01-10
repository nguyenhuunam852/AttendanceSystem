import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity


class FaceMonitor:
    def __init__(self, id, features, cords):
        self.features = features
        self.id = id
        self.cord = cords


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def most_similarity(embed_vec, faces, threshhold=0.8):
    vecs = np.array([x.features for x in faces])
    sim = cosine_similarity(np.array([embed_vec]), vecs)
    temp = np.sort(sim[0])[::-1]
    if temp[0] < threshhold:
        return -1
    else:
        argmax = np.argsort(sim[0])[::-1]
        return argmax[0]


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


def create_box_encoder(model_filename, input_name="images", output_name="Identity", batch_size=32):
    image_encoder = ImageEncoder(model_filename, input_name, output_name)

    def encoder(images):
        image_patches = np.asarray(images)
        return image_encoder(image_patches, batch_size)

    return encoder


class Embedding:
    def __init__(self, model):
        self.encoder = create_box_encoder(model, batch_size=32)
        self.prev_embed = None
        self.faces = None

    def extract_face(self, image, bboxes):
        emb_faces = []
        for index, item in enumerate(bboxes):
            emb_face = image[int(item[0]):int(item[2]),
                             int(item[1]):int(item[3])]
            emb_face = cv2.resize(emb_face, (112, 112))
            emb_face = emb_face.astype(np.float32) / 255.
            # emb_face = cv2.cvtColor(emb_face, cv2.COLOR_BGR2RGB)
            cv2.imwrite('test{0}.png'.format(str(index)), emb_face)
            emb_faces.append(emb_face)

        emb_faces = np.array(emb_faces)

        with tf.device('/GPU:0'):
            features = self.encoder(emb_faces)

        features = l2_norm(features)

        return features
