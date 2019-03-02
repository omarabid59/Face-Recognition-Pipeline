import ..helper_nn_models_utils as nn_model_utils
import threading
from abc import ABC,  abstractmethod

from aipodMain.utilities.ObjectTracking.old_tracker.main_tracker import GlobalTracker

import tensorflow as tf
import numpy as np
from ..DataHolder import ClassificationData
import cv2

class AbstractPredictor(threading.Thread, ABC):
    def __init__(self, name,
                 PATH_TO_LABELS,
                 IMG_SCALE,
                 TRACKER_TYPE = None,
                 ENABLE_BY_DEFAULT=False):
        """
        TRACKER_TYPE: ['Legacy', 'DeepSort']
        """
        self.TRACKER_DEEPSORT = 'DeepSort'
        self.TRACKER_LEGACY = 'Legacy'

        threading.Thread.__init__(self)
        ABC.__init__(self)
        self.name = name
        self.done = False
        self.pause = not ENABLE_BY_DEFAULT

        self.IMG_SCALE = IMG_SCALE

        # TRACKER
        self.TRACKER_TYPE = TRACKER_TYPE
        if TRACKER_TYPE == self.TRACKER_LEGACY:
            self.global_tracker = GlobalTracker()
        elif TRACKER_TYPE == self.TRACKER_DEEPSORT:
            self.global_tracker = DeepSortTracker()
            self.global_tracker.min_confidence = 0.1
            self.global_tracker.nms_max_overlap = 1
        elif TRACKER_TYPE == None:
            print('Tracker Disabled.')
        else:
            assert False, "Error. Invalid tracker type defined."




        [self.category_index, self.NUM_CLASSES] = nn_model_utils.get_label_map(PATH_TO_LABELS)

    def setLegacyTrackerParameters(self,MIN_HITS,MAX_AGE):
        self.global_tracker.min_hits = MIN_HITS
        self.global_tracker.max_age = MAX_AGE

    def setDeepSortTrackerParameters(self,
                                     max_age = 3,
                                    n_init = 3,
                                     nms_max_overlap =1.0,
                                    min_detection_height = 0.0, nn_budget =None,
                                     max_cosine_distance = 0.2):
        """
        Parameters
        ----------
        max_age : int
            Maximum number of missed misses before a track is deleted.
        n_init : int
            Number of consecutive detections before the track is confirmed. The
            track state is set to `Deleted` if a miss occurs within the first
            `n_init` frames.
        nms_max_overlap: float
            Maximum detection overlap (non-maxima suppression threshold).
        min_detection_height : int
            Detection height threshold. Disregard all detections that have
            a height lower than this value.
        max_cosine_distance : float
            Gating threshold for cosine distance metric (object appearance).
        nn_budget : Optional[int]
            Maximum size of the appearance descriptor gallery. If None, no budget
            is enforced.
        """
        # Restabilish the tracker
        self.global_tracker = DeepSortTracker(self.output_data.score_thresh,
                 nms_max_overlap,
                 min_detection_height,
                 nn_budget,
                max_cosine_distance,
                max_age,
                n_init)
        # Use a new type of data frame to encapsulate the data
        score_threshold = self.output_data.score_thresh
        category_index = self.output_data.category_index
        self.output_data = OutputClassificationDataWithTracker()
        self.output_data.category_index = category_index
        self.output_data.score_thresh = score_threshold
        print("Using OutputClassificationDataWithTracker() to encapsulate data")
        print("WARNING. Abstract Predictor has no check in place to ensure score threshold is defined. Manual check required.")





    def load_model(self, PATH_TO_MODEL, GPU_ALLOCATION = 1.0):
        print('Loading Model File: ' + PATH_TO_MODEL)
        if GPU_ALLOCATION < 1.0:
            print("Alocating " + str(GPU_ALLOCATION*100) + '% of GPU')
        # Load the graphs.
        [graph, sess] = nn_model_utils.load_graph_with_sess(PATH_TO_MODEL, GPU_ALLOCATION)
        print('Finished Loading Model')
        return [graph, sess]

    def run(self):
        print("Starting " + self.name)
        self.predict(self.name)
        print("Exiting " + self.name)
    def pause_predictor(self):
        self.pause = True
    def continue_predictor(self):
        self.pause = False
    def stop(self):
        self.done = True

    @abstractmethod
    def predict(self,threadName):
        pass
    @abstractmethod
    def predict_once(self,image_np):
        pass

    def getImage(self):
        '''
        Returns the resized image that we will use for prediction.
        '''
        self.output_data.image_data.original_image_np = self.image_data.image_np
        if self.IMG_SCALE < 1.0:
            self.output_data.image_data.image_np =  cv2.resize(self.image_data.image_np.copy(), (0,0), fx=self.IMG_SCALE, fy=self.IMG_SCALE)
        else:
            self.output_data.image_data.image_np = self.image_data.image_np
        return self.output_data.image_data.image_np
