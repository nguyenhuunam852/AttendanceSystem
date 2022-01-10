# %%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from faces_module import Embedding
from modules.models import ArcFaceModel
from centernet import Detector
import cv2
import numpy as np
import glob
from sklearn.metrics.pairwise import cosine_similarity
# %%
model = ArcFaceModel(size=112,
                     backbone_type='ResNet50',
                     training=False)
ckpt_path = tf.train.latest_checkpoint('./arc_res50_qmuls/')
model.load_weights(ckpt_path)
model.save('./save_model5/')
# %%
model = tf.keras.models.load_model('save_model', compile=False)
# %%
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="images"))
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 50)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 50)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph from frozen ConcreteFunction to hard drive
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph.pb",
                  as_text=False)
# %%
PATH_TO_CFG = r'D:\train2017\KhoaLuanTotNghiep\My_MLApp\config\pipeline.config'
PATH_TO_CKPT = r'D:\train2017\KhoaLuanTotNghiep\My_MLApp\Centernet-2192021-1147-faces/ckpt-108'
PATH_TO_LABELS = r'D:\train2017\KhoaLuanTotNghiep\My_MLApp\config\label_map.txt'
PATH_TO_TRACKING = r'D:\train2017\KhoaLuanTotNghiep\RarFile\arcface-tf2-master\frozen_models5\frozen_graph.pb'
# %%
face_emb = Embedding(
    r'D:\train2017\KhoaLuanTotNghiep\RarFile\arcface-tf2-master\frozen_models1\frozen_graph.pb')
detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)

# %%
pictures = glob.glob(
    r'D:\train2017\KhoaLuanTotNghiep\RarFile\arcface-tf2-master\pictures1\*')

sface_emb = []
for item in pictures:
    frame = cv2.imread(item)
    frame = cv2.resize(frame, (512, 512))
    out_scores, classes, detections = detector.predict(frame)

    detections = np.array(detections)
    out_scores = np.array(out_scores)

    set_detection = []
    set_out_scores = []

    for index, item in enumerate(detections):
        if out_scores[index] > 0.5:
            min_point = item[:2]
            min_point = min_point * (512, 512)
            max_point = item[2:]
            max_point = max_point * (512, 512)
            set_detection.append(
                np.concatenate([min_point, max_point]))

    set_detection = np.array(set_detection)

    if len(set_detection) != 0:
        faces = face_emb.extract_face(frame, set_detection)

    sface_emb.append(faces)

test = cosine_similarity(sface_emb[0], sface_emb[1])
test1 = cosine_similarity(sface_emb[1], sface_emb[2])
# %%
