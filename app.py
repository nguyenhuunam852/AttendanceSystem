# activate lib
# import keras
# import keras.api
# import keras.api._v1
# import keras.api._v2
# import keras.engine.base_layer_v1

#
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import numpy as np
from centernet import Detector
from deepsort import deepsort_rbc
import uuid
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from Faces_Module.Face import FaceMonitor, FeatureMonitor


def _save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def _load_pickle(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


PATH_TO_CFG = 'config//pipeline.config'
PATH_TO_CKPT = 'Centernet-9242021-1551-faces//ckpt-85'
PATH_TO_LABELS = 'config//label_map.txt'
PATH_TO_TRACKING = 'frozen_models//frozen_graph.pb'

detector = Detector(PATH_TO_CFG, PATH_TO_CKPT, PATH_TO_LABELS)
deep_sort = deepsort_rbc(PATH_TO_TRACKING)
frame_id = 0
dict_person = {}
list_face_cache = []

cache_faces = None


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def my_random_string(string_length=10):
    random = str(uuid.uuid4())
    random = random.upper()
    random = random.replace("-", "")
    return random[0:string_length]


def consine_distance(emb_vec, vecs, check=True, maxthreshhold=0.7, minthreshhold=0.3):
    sim = cosine_similarity(np.array([emb_vec]), vecs)
    sort_sim = np.sort(sim[0])[::-1]
    if(check is False):
        print(sim)
    if (max(sort_sim) > maxthreshhold or max(sort_sim) < minthreshhold and check is True):
        return True
    return False


def extract_face(frame, bbox, id_num):
    face = frame[int(bbox[0]):int(
        bbox[2]), int(bbox[1]):int(bbox[3])]
    faceshape = face.shape
    if all(face.shape):
        face = cv2.cvtColor(
            face, cv2.COLOR_RGB2BGR)
        face = cv2.resize(face, (112, 112))
        item = QListWidgetItem(id_num.split('_')[0])
        icon = QtGui.QIcon()
        image = QtGui.QImage(
            face.data, face.shape[1], face.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        icon.addPixmap(QtGui.QPixmap.fromImage(image))
        item.setIcon(icon)
        return item, face, faceshape
    return None, None, None


def save_emb(parent, selectid, itemid):
    global dict_person
    global list_face_cache

    xemb = next((x for x in list_face_cache if x.label == selectid), None)

    listemb = [x.feature for x in xemb.features]
    get_features = np.array(listemb)

    # name = [x.featureid for x in xemb.features]
    # for item in name:
    #     print(item)

    if xemb is not None:
        for index, item in enumerate(list_face_cache):
            if item.label == xemb.label:
                for face in dict_person[itemid]['emb']:
                    if(consine_distance(face.feature, get_features, False) is not True):
                        list_face_cache[index].features.append(face)
                        parent.tab_widget.createListFaceManager(
                            list_face_cache[index], False)

    get_list = parent.listitem.findItems(
        itemid, Qt.MatchExactly)

    for item in get_list:
        parent.listitem.takeItem(
            parent.listitem.row(item))

    listemb = []
    listlabel = []
    listfid = []
    listpicture = []

    for person in list_face_cache:
        for emb in person.features:
            listlabel.append(person.label)
            listemb.append(emb.feature)
            listpicture.append(emb.face)
            listfid.append(emb.featureid)

    _save_pickle(listemb, "./pickle/list_faces.pkl")
    _save_pickle(listlabel, "./pickle/list_labels.pkl")
    _save_pickle(listfid, "./pickle/list_pictures_id.pkl")
    _save_pickle(listpicture, "./pickle/list_pictures.pkl")

    print('saved label:{0}'.format(str(len(listlabel))))
    print('saved emb:{0}'.format(str(len(listemb))))


def _most_similarity(parent, listface, vec, item, vec_label, minthreshhold=0.7):
    max_dict = {}

    try:
        for face in listface:
            face_list = [x.feature for x in face.features]
            sim = cosine_similarity(vec, face_list)
            sim = np.sort(sim)[:, ::-1]
            check = sim[:, 0]
            max_value = max(check)
            index = np.argmax(check)
            if max_value < minthreshhold:
                continue
            save_emb(parent, face.label, vec_label[index].split('_')[0])
            print(str(item) + ':' + str(max_value) +
                  '-kt:' + face.label + '-test:' + vec_label[index])
            max_dict[face.label] = max_value
    except:
        print('error from vec')

    if len(max_dict.keys()):
        return max(max_dict, key=max_dict.get)
    return 0.0


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, detect, parent, listemb, listlabel):
        super().__init__()
        self.detector = detect
        # rtsp://admin:nam781999@192.168.1.206:554/cam/realmonitor?channel=1&subtype=0
        self.sparent = parent
        self.list_emb = listemb
        self.list_label = listlabel
        self.video = cv2.VideoCapture(
            r'D:\ComputerVisionProject\ComputerVision\cVideo\Pier Park Panama City_ Hour of Watching People Walk By.mp4')

    def update_dict(self, tracker, exist):
        for item in dict_person.keys():
            dict_person[item]['appearance'] += 1

        if exist:
            get_list_deleted = [x for x in dict_person.keys(
            ) if x not in [y.track_id for y in tracker.tracks] and dict_person[x]['appearance'] > 30
                and dict_person[x]['active']]
        else:
            get_list_deleted = [x for x in dict_person.keys(
            ) if dict_person[x]['appearance'] > 30 and dict_person[x]['active']]

        for item in get_list_deleted:
            vec = dict_person[item]
            dict_person[item]['appearance'] = 0
            dict_person[item]['active'] = False
            vec = [x.feature for x in dict_person[item]['emb']]
            vec_label = [
                x.featureid for x in dict_person[item]['emb']]

            if len(self.list_emb) != 0:
                _most_similarity(
                    self.sparent, list_face_cache, vec, item, vec_label)

    def run(self):
        global list_face_cache

        frameid = 0
        while self.video.isOpened():
            ret, frame = self.video.read()
            if ret:
                if frameid == 4:
                    frame = cv2.resize(frame, (512, 512))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    frameid = 0
                    height, width, _ = frame.shape

                    set_detection, set_out_scores = detector.predict(
                        frame, height, width, 0.3)

                    if len(set_detection) != 0:
                        tracker, detections_class = deep_sort.run_deep_sort(
                            frame, set_out_scores, set_detection)

                        self.update_dict(tracker, True)

                        if tracker is not None:
                            for track in tracker.tracks:
                                if not track.is_confirmed() or track.time_since_update > 1:
                                    continue

                                bbox = track.to_tlbr()
                                id_num = str(track.track_id)
                                features = track.features

                                wh_box = bbox[2:] - bbox[:2]

                                # wh_box[0] > 50 or wh_box[1] > 50 or

                                if wh_box[0] > 30 or wh_box[1] > 30 or wh_box[0] < 10 or wh_box[1] < 10:
                                    self.change_pixmap_signal.emit(frame)
                                    continue

                                if len(bbox) == 0:
                                    self.change_pixmap_signal.emit(frame)
                                    continue

                                if id_num not in dict_person.keys():
                                    dict_person[id_num] = {}
                                    dict_person[id_num]['emb'] = []
                                    dict_person[id_num]['appearance'] = 0
                                    dict_person[id_num]['active'] = True

                                    rid = id_num + '_' + my_random_string(6)
                                    picture, face, shapef = extract_face(
                                        frame, bbox, rid)
                                    if shapef is None:
                                        continue
                                    feature = FeatureMonitor(
                                        rid+"_{0}x{1}".format(str(shapef[0]), str(shapef[1])), features[0], face)
                                    self.sparent.listitem.addItem(picture)
                                    dict_person[id_num]['emb'].append(feature)

                                else:
                                    emb_vec = features[0]
                                    dict_person[id_num]['appearance'] = 0
                                    listemb = [
                                        x.feature for x in dict_person[id_num]['emb']]

                                    get_features = np.array(listemb)
                                    check = consine_distance(
                                        emb_vec, get_features)

                                    if check:
                                        continue

                                    rid = id_num + '_' + \
                                        my_random_string(6)

                                    picture, face, shapef = extract_face(
                                        frame, bbox, rid)

                                    feature = FeatureMonitor(
                                        rid+"_{0}x{1}".format(str(shapef[0]), str(shapef[1])), features[0], face)

                                    get_list = self.sparent.listitem.findItems(
                                        id_num, Qt.MatchExactly)

                                    for item in get_list:
                                        self.sparent.listitem.takeItem(
                                            self.sparent.listitem.row(item))

                                    self.sparent.listitem.addItem(picture)
                                    dict_person[id_num]['emb'].append(
                                        feature)

                                cv2.rectangle(frame, (int(bbox[1]), int(bbox[0])), (int(
                                    bbox[3]), int(bbox[2])), (255, 255, 255), 2)

                                cv2.putText(frame, str(id_num), (int(bbox[1]), int(
                                    bbox[0])), 0, 5e-3 * 200, (0, 255, 0), 2)

                    else:
                        self.update_dict(None, False)

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.change_pixmap_signal.emit(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                else:
                    frameid += 1

        self.video.release()


class MyTabWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()

        self.tabs.addTab(self.tab1, "Main")
        self.tabs.addTab(self.tab2, "Manage")

        self.tab1.layout = QHBoxLayout(self)
        self.tab2.layout = QVBoxLayout(self)

        self.tab1.layout.addWidget(parent.image_label, 50)
        self.tab1.layout.addLayout(parent.rightpanel, 50)

        self.qlistwiget = {}

        self.button_delete_peson = QPushButton('Delete', self)
        self.button_delete_peson.clicked.connect(self.on_delete_person)
        self.tab2.layout.addWidget(self.button_delete_peson)

        for item in list_face_cache:
            self.createListFaceManager(item, True)

        self.tab1.setLayout(self.tab1.layout)
        self.tab2.setLayout(self.tab2.layout)

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def createListFaceManager(self, item1, signal):
        if signal:
            item_label = QLabel(item1.label)
            item_label.setStyleSheet("background-color: lightgreen")
            self.tab2.layout.addWidget(item_label)

            self.qlistwiget[item1.label] = QListWidget(self)
            self.qlistwiget[item1.label].setViewMode(
                QtWidgets.QListView.IconMode)
            self.qlistwiget[item1.label].setIconSize(QtCore.QSize(112, 112))
            self.qlistwiget[item1.label].setResizeMode(
                QtWidgets.QListView.Adjust)
            self.qlistwiget[item1.label].setSelectionMode(
                QtWidgets.QListView.ExtendedSelection)

            for feature in item1.features:
                if all(feature.face.shape):
                    face = cv2.resize(feature.face, (112, 112))
                    item = QListWidgetItem(feature.featureid)
                    icon = QtGui.QIcon()
                    image = QtGui.QImage(
                        face.data, face.shape[1], face.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image))
                    item.setIcon(icon)
                    self.qlistwiget[item1.label].addItem(item)

            self.tab2.layout.addWidget(self.qlistwiget[item1.label])

        else:
            self.qlistwiget[item1.label].clear()

            for feature in item1.features:
                if all(feature.face.shape):
                    face = cv2.resize(feature.face, (112, 112))
                    item = QListWidgetItem(feature.featureid)
                    icon = QtGui.QIcon()
                    image = QtGui.QImage(
                        face.data, face.shape[1], face.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
                    icon.addPixmap(QtGui.QPixmap.fromImage(image))
                    item.setIcon(icon)
                    self.qlistwiget[item1.label].addItem(item)

    @pyqtSlot()
    def on_delete_person(self):
        global list_face_cache

        for item in self.qlistwiget.keys():
            select_index = [
                index for index, x in enumerate(list_face_cache) if x.label == item][0]
            items = self.qlistwiget[item].selectedItems()
            for picture in items:
                self.qlistwiget[item].takeItem(
                    self.qlistwiget[item].row(picture))
                select_feature_index = [
                    index for index, x in enumerate(list_face_cache[select_index].features) if
                    x.featureid == picture.text()][0]
                del list_face_cache[select_index].features[select_feature_index]

        listemb = []
        listlabel = []
        listfid = []
        listpicture = []

        for person in list_face_cache:
            for emb in person.features:
                listlabel.append(person.label)
                listemb.append(emb.feature)
                listpicture.append(emb.face)
                listfid.append(emb.featureid)

        _save_pickle(listemb, "./pickle/list_faces.pkl")
        _save_pickle(listlabel, "./pickle/list_labels.pkl")
        _save_pickle(listfid, "./pickle/list_pictures_id.pkl")
        _save_pickle(listpicture, "./pickle/list_pictures.pkl")


def loadFeatures(listemb, listlabel, listpicture, listfeatureid):
    listfacecache = []
    for index, item in enumerate(listlabel):
        checklist = [index for index, x in enumerate(
            listfacecache) if x.label == item]
        if len(checklist) == 0:
            face = FaceMonitor(item, True)
            feature = FeatureMonitor(
                listfeatureid[index], listemb[index], listpicture[index])
            face.features.append(feature)
            listfacecache.append(face)
        else:
            feature = FeatureMonitor(
                listfeatureid[index], listemb[index], listpicture[index])
            listfacecache[checklist[0]].features.append(feature)
    return listfacecache


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        global list_face_cache

        listemb = None
        listlabel = None
        listpicture = None
        listfeatures = None

        check = os.path.isfile('./pickle/list_faces.pkl')
        label = os.path.isfile('./pickle/list_labels.pkl')
        picture = os.path.isfile('./pickle/list_pictures.pkl')
        features = os.path.isfile('./pickle/list_pictures_id.pkl')

        if check and label and picture and features:
            listemb = _load_pickle('./pickle/list_faces.pkl')
            listlabel = _load_pickle('./pickle/list_labels.pkl')
            listpicture = _load_pickle('./pickle/list_pictures.pkl')
            listfeatures = _load_pickle('./pickle/list_pictures_id.pkl')

        else:
            listemb = []
            listlabel = []
            listpicture = []
            listfeatures = []

        list_face_cache = loadFeatures(
            listemb, listlabel, listpicture, listfeatures)

        self.left = 0
        self.top = 0
        self.width = 1024
        self.height = 512
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.display_width = 512
        self.display_height = 512

        self.image_label = QLabel(self)
        self.image_label.setStyleSheet(
            "background: gray;border: 1px solid black;")

        self.listitem = QListWidget(self)
        self.listitem.setViewMode(QtWidgets.QListView.IconMode)
        self.listitem.setIconSize(QtCore.QSize(112, 112))
        self.listitem.setResizeMode(QtWidgets.QListView.Adjust)
        self.listitem.setSelectionMode(
            QtWidgets.QListView.ExtendedSelection)

        self.combo = QComboBox(self)
        self.combo.addItem('Người A')
        self.combo.addItem('Người B')
        self.combo.addItem('Người C')
        self.combo.addItem('Người D')

        self.listbuton = QHBoxLayout(self)
        self.button_accept = QPushButton('Done', self)
        self.button_accept.clicked.connect(self.on_click)
        self.button_delete = QPushButton('Delete', self)
        self.button_delete.clicked.connect(self.on_delete)

        self.listbuton.addWidget(self.button_accept, 50)
        self.listbuton.addWidget(self.button_delete, 50)

        self.rightpanel = QVBoxLayout(self)
        self.rightpanel.addWidget(self.combo, 10)
        self.rightpanel.addWidget(self.listitem, 80)
        self.rightpanel.addLayout(self.listbuton)

        self.tab_widget = MyTabWidget(self)
        self.setCentralWidget(self.tab_widget)

        self.thread = VideoThread(
            detector, self, listemb, listlabel)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    @pyqtSlot()
    def on_click(self):
        global dict_person
        global list_face_cache
        # print(len(list_face_cache[0].features))

        selectid = self.combo.currentText()
        items = self.listitem.selectedItems()
        list_id = [x.text().split('_')[0] for x in items]
        list_id = list(dict.fromkeys(list_id))

        x = next((x.label for x in list_face_cache if x.label == selectid), None)

        if x is not None:
            for index, item in enumerate(list_face_cache):
                if item.label == x:
                    for fid in list_id:
                        for face in dict_person[fid]['emb']:
                            list_face_cache[index].features.append(face)
                    self.tab_widget.createListFaceManager(
                        list_face_cache[index], False)
        else:
            facem = FaceMonitor(selectid, True)
            for fid in list_id:
                for face in dict_person[fid]['emb']:
                    facem.features.append(face)
            list_face_cache.append(facem)
            self.tab_widget.createListFaceManager(
                list_face_cache[len(list_face_cache) - 1], True)

        items = self.listitem.selectedItems()
        for item in items:
            self.listitem.takeItem(
                self.listitem.row(item))

        listemb = []
        listlabel = []
        listfid = []
        listpicture = []

        for person in list_face_cache:
            for emb in person.features:
                listlabel.append(person.label)
                listemb.append(emb.feature)
                listpicture.append(emb.face)
                listfid.append(emb.featureid)

        _save_pickle(listemb, "./pickle/list_faces.pkl")
        _save_pickle(listlabel, "./pickle/list_labels.pkl")
        _save_pickle(listfid, "./pickle/list_pictures_id.pkl")
        _save_pickle(listpicture, "./pickle/list_pictures.pkl")

        print('saved label:{0}'.format(str(len(listlabel))))
        print('saved emb:{0}'.format(str(len(listemb))))

    @pyqtSlot()
    def on_delete(self):
        items = self.listitem.selectedItems()
        for item in items:
            self.listitem.takeItem(
                self.listitem.row(item))

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(
            self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
