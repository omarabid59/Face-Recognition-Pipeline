from __future__ import division
import ..helper_nn_models_utils as nn_model_utils
from .AbstractPredictor import AbstractPredictor
import numpy as np
import sys
import align.detect_face
import facenet
import tensorflow as tf
from ..DataHolder import DetectionData
import time

class FacePredictor(threading.Thread):
    def __init__(self,PATH_TO_LABELS,
                 image_data,
                  score_thresh,
                 IMG_SCALE):


        name = "MTCNN Face Predictor"
        super().__init__(name,
                         PATH_TO_LABELS,
                         IMG_SCALE)


        self.image_data = image_data
        self.output_data = DetectionData()
        self.output_data.score_thresh = score_thresh
        self.output_data.category_index = self.category_index
        self.load_model()
        self.PREDICT_INTERVAL_MS = 20

    def load_model(self):
        print("Loading MTCNN Model")
        gpu_memory_fraction = 0.1
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)
        print("MTCNN. Setting parameters.")
        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
        self.factor = 0.709 # scale factor
        self.margin = 32
        self.image_size = 160
        self.detect_multiple_faces = True

    def predict_once(self,image_np):
        tracker_ids = []
        # BGR -> RGB
        image_np = image_np[:,:,::-1]
        if image_np.ndim == 2:
            image_np = facenet.to_rgb(image_np)
        image_np = image_np[:,:,0:3]

        bounding_boxes, _ = align.detect_face.detect_face(image_np, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces>0:
            tracker_ids = []
            det = bounding_boxes[:,0:4]
            SCORES = bounding_boxes[:,-1]
            det_arr = []
            img_size = np.asarray(image_np.shape)[0:2]
            if nrof_faces>1:
                if self.detect_multiple_faces:
                    for i in range(nrof_faces):
                        det_arr.append(np.squeeze(det[i]))
                else:
                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                    img_center = img_size / 2
                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                    det_arr.append(det[index,:])
            else:
                det_arr.append(np.squeeze(det))
            boxes = np.zeros(shape=(len(det_arr),4),dtype=np.float32)
            scores = np.zeros(shape=(len(det_arr),))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.float32)

                bb[1] = det[0]
                bb[0] = det[1]
                bb[3] = det[2]
                bb[2] = det[3]

                bb[0] = float(bb[0]) / float(img_size[0])
                bb[1] = float(bb[1]) / float(img_size[1])
                bb[2] = float(bb[2]) / float(img_size[0])
                bb[3] = float(bb[3]) / float(img_size[1])


                boxes[i,:] = bb
                scores[i] = SCORES[i]



            # Eliminate all values that do not meet the threshold.
            indices = [i for i, x in enumerate(scores > self.output_data.score_thresh) if x]
            scores = scores[indices]
            boxes = boxes[indices]
            # Run the tracker
            if self.ENABLE_TRACKER:
                [tracker_ids, boxes, scores, _] = self.runTracker(boxes,scores,
                                                                image_np)
            self.output_data.bbs = boxes
            self.output_data.scores = scores
            self.output_data.tracker_ids = tracker_ids
            self.output_data.image_data.image_np = image_np


        else:
            self.output_data.bbs = np.asarray([])
        time.sleep(self.PREDICT_INTERVAL_MS/1000.0)

        #########################################


    def runTracker(self, boxes,scores, image_np):
        [tracker_ids, boxes,scores, _,
                    _] = self.global_tracker.pipeline(boxes,
                                                      scores,
                                                      scores,
                                                      image_np, return_tracker_id = True)
        return [tracker_ids, boxes,scores, _]

    def predict(self,threadName):
        while not self.done:
            image_np = self.getImage()
            self.predict_once(image_np)
