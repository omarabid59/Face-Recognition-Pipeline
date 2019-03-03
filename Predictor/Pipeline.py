import copy
import threading
import time
from .Detector import FacePredictor
from .Recognizer import FaceRecognition
import ..constants as constants
from ..helper_data_encapsulation import OutputFaceData
from .HistoryAveraging import HistoryAveraging
class FaceDTCM():

    def __init__(self,image_data,
                 SVM_CLASSIFIER_PATH,
                 NN_CLASSIFIER_PATH,
                 detector_thresh=0.6,
                 recognition_thresh=0.6,
                DETECTOR_IMG_SCALE = 1.0,
                AUTO_START=True):
        """
        DETECTOR_IMG_SCALE:
            Sets the scale size of the image for the detection process.
        """


        # Encapsulate both the detection and recognition data so it is easier to deal with.
        self.output_data = OutputFaceData()




        self.name = "thread_frm"
        self.isStarted = False

        self.thread_detector = FacePredictor(image_data,
                                    detector_thresh,
                                     IMG_SCALE = DETECTOR_IMG_SCALE,
                                    ENABLE_TRACKER = ENABLE_TRACKER)


        self.thread_recognition = FaceRecognition(self.output_data,
                                                 SVM_CLASSIFIER_PATH,
                                                 NN_CLASSIFIER_PATH)

        self.history_average = HistoryAveraging()

        self.output_data.detection_data = self.thread_detector.output_data;
        # Set default thresholds.
        self.setThresholds(detector_thresh,recognition_thresh)
        if AUTO_START:
            startAll();

    def startAll():
        if self.isStarted == False:
            self.thread_detector.start()
            self.thread_recognition.start()
            self.thread_detector.continue_predictor()
            self.thread_recognition.continue_predictor()
            self.isStarted = True
    def setThresholds(self, detector_thresh=0.6, recognition_thresh=0.6):
        """
        detector_thresh:
            Threshold for our MTCNN mobilenet detection of faces.
        recognition_thresh:
            Threshold for classification.
        """
        self.output_data.detection_data.score_thresh = detector_thresh
        self.output_data.recognition_data.score_thresh = recognition_thresh

    def setUpdateInterval(self, detector_ms = 100,recognition_ms = 20):
        """
        detector_ms:
            Face detection.
        recognition_ms:
            Face Recognition
        """
        self.thread_detector.update_interval_ms = detector_ms
        self.thread_recognition.update_interval_ms = recognition_ms


    def stop(self):
        self.thread_detector.stop()
        self.thread_recognition.stop()
    def detect_and_recognize(with_history=True):
        '''
        TODO: Return the result of detection and recognition
        with_history: Returns the results using the history of previously recognized faces.
        '''
        if with_history:
            results = history_average.detect_and_recognize_results()
        return results
    def detect_only():
        '''
        TODO: Return the result of detection only.
        '''
        pass
    def recognize_only(detection_data):
        '''
        TODO: Return the result of recognition given detection data
        '''
        pass
