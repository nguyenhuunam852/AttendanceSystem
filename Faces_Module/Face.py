class FaceMonitor:
    def __init__(self, label, signal, picturelist=None):
        self.features = []
        self.label = label
        self.visible = signal
        self.picturelist = picturelist


class FeatureMonitor:
    def __init__(self, featureid, feature, face):
        self.feature = feature
        self.featureid = featureid
        self.face = face
