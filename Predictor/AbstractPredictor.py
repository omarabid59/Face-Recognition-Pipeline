import ..helper_nn_models_utils as nn_model_utils
import threading
from abc import ABC,  abstractmethod
from ..Tracker.main_tracker import GlobalTracker

import tensorflow as tf
import numpy as np
from ..DataHolder import ClassificationData
import cv2

class AbstractPredictor(threading.Thread, ABC):
    def __init__(self, name,
                 PATH_TO_LABELS,
                 IMG_SCALE):


        threading.Thread.__init__(self)
        ABC.__init__(self)
        self.name = name
        self.done = False
        self.pause = not ENABLE_BY_DEFAULT

        self.IMG_SCALE = IMG_SCALE

        self.global_tracker = GlobalTracker()
        self.output_data = None


        [self.category_index, self.NUM_CLASSES] = nn_model_utils.get_label_map(PATH_TO_LABELS)

    def setLegacyTrackerParameters(self,MIN_HITS,MAX_AGE):
        self.global_tracker.min_hits = MIN_HITS
        self.global_tracker.max_age = MAX_AGE




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
