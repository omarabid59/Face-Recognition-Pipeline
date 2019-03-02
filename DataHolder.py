import numpy as np
class FaceData:
    def __init__(self):
        self.detection_data = None#self.DetectionData()
        self.recognition_data = None#self.RecognitionData()
        self.results_data = None#[]

class DetectionData():
    def __init__(self):
        self.tracker_ids = np.asarray([])
        self.bbs = np.asarray([])
        self.scores = np.asarray([])
        self.score_thresh = 0.0
        self.category_index = ()
        self.image_data = ImageData()
class RecognitionData():
    def __init__(self):
        self.score_thresh = 0.0
        self.history = self.History()
        self.intermediate_data = self.IntermediateData()
        self.output_data = self.Results()
        self.EMPTY_ELEMENT = 'frm:unknown'
        self.EMPTY_MAJORITY_ELEMENTS = [self.EMPTY_ELEMENT] * self.history.RECOGNITION_HISTORY_LENGTH
    class IntermediateData():
        def __init__(self):
            self.scores = np.asarray([])
            self.classes = np.asarray([])
            self.tracker_ids = []

    class History():
        def __init__(self):
            # An (N,RECOGNITION_HISTORY_LENGTH) list. Stores the previous recognition results for improved accuracy.
            self.recognition_history = []
            self.idle_age = []
            self.RECOGNITION_HISTORY_LENGTH = 5
            self.tracker_ids = []
            self.MAX_AGE = 5
            self.persons = []
            self.scores = []
            self.age = []
    class Results():
        def __init__(self):
            """
            Stores the current results we wish to use.
            """
            self.tracker_ids = []
            self.persons = []
            self.scores = []



class ImageData:
    def __init__(self):
        self.TYPE = 'Webcam'
        self.image_np = ()
        self.isInit = False
        self.width = None
        self.height = None

'''
class ClassificationData:
    def __init__(self):
        self.bbs = np.asarray([])
        self.scores = np.asarray([])
        self.classes = np.asarray([])
        self.image_data = ImageData()
        self.category_index = ()
        self.tracker_ids = np.asarray([])
        self.tracked_bbs = np.asarray([])
'''
