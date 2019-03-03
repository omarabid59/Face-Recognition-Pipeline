import numpy as np
class FaceData:
    def __init__(self):
        self.detection_data = self.DetectionData()
        self.recognition_data = self.RecognitionData()
        self.results_data = self.ResultData()

    class DetectionData():
        def __init__(self):
            self.tracker_ids = np.asarray([])
            self.bbs = np.asarray([])
            self.scores = np.asarray([])
            self.score_thresh = 0.0

    class RecognitionData():
        def __init__(self):
            self.score_thresh = 0.0
            self.scores = np.asarray([])
            self.persons = np.asarray([])
            self.tracker_ids = []

    class ResultData:
        def __init__(self):
            self.bbs = []
            self.persons = []
            self.scores = []

class ImageData:
    def __init__(self):
        self.TYPE = 'Webcam'
        self.image_np = ()
        self.isInit = False
        self.width = None
        self.height = None
