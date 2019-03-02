from .AbstractSubsetImgPredictor import AbstractSubsetImgPredictor

import pickle
import threading
import time
import warnings
import facenet
import cv2
import ..constants as constants
import ..helper_nn_models_utils as nn_model_utils
import numpy as np
class FaceRecognition(AbstractSubsetImgPredictor):
    def __init__(self, face_data,
                SVM_CLASSIFIER_PATH,NN_CLASSIFIER_PATH):
        """
        global_tracker:
            Rather than passing the `output_data` frame, we pass the `global_tracker` since this contains both the
            bounding boxes and the corresponding tracking ID.
        detection_data:
            For now we simply pass the output_data. We only need the "detection_data however."
        """
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        threading.Thread.__init__(self)
        self.done = False
        self.name = "Face Recognition Thread."

        self.output_data = face_data


        self.net = self.FaceNetModel(SVM_CLASSIFIER_PATH,NN_CLASSIFIER_PATH)

        self.UPDATE_INTERVAL_MS = 20



    def get_faces_w_margin(self,bbs,image_np):
        '''
        Gets the subset image containing faces given the bounding box coordinates in the image image_np.
        '''
        h,w,_ = image_np.shape
        faces = []
        for bb in bbs:
            width  = (bb[3] - bb[1])*w
            height = (bb[2] - bb[0])*h
            margin = int(min(height,width)*0.125)
            y = int(np.maximum(int(h*bb[0]) - margin,0))
            x = int(np.maximum(int(w*bb[1]) - margin,0))
            y_ = int(np.minimum(int(h*bb[2]) + margin,h))
            x_ = int(np.minimum(int(w*bb[3]) + margin,w))
            if x > x_ or y > y_:
                continue
            sub_img = image_np[int(y):int(y_),
                               int(x):int(x_),:]

            faces.append(sub_img)
        return faces



    def predict_once(self):
        detection_data = self.output_data.detection_data
        image_np = detection_data.image_data.original_image_np
        bbs = list(detection_data.bbs)
        tracker_ids = list(detection_data.tracker_ids)
        # Extract all of the faces
        faces = self.get_faces_w_margin(bbs, image_np)
        persons = []
        scores = []
        class_index = []
        for face in faces:
            [best_class_indices,best_class_probabilities] = self.net.get_classification(face)
            scores.append(best_class_probabilities[0])
            persons.append(self.net.clf_class_names[best_class_indices[0]])
        # Remove all of those that do not meet the threshold criteria.
        for indx,score in enumerate(scores):
            if score < self.output_data.recognition_data.score_thresh:
                persons[indx] = ''
        return (persons, scores, tracker_ids)

    def predict(self, threadName):
        intermediate_data = self.output_data.recognition_data.intermediate_data
        counter = 0
        while not self.done:
            if not self.pause:
                classes, scores ,tracker_ids = self.predict_once()
                time.sleep(self.UPDATE_INTERVAL_MS/1000.0)
            else:
                scores = np.asarray([]).reshape(-1,1)
                classes = []
                tracker_ids = []
                time.sleep(1.0)

            intermediate_data.scores = np.asarray(list(scores))
            intermediate_data.classes = list(['frm:' + s for s in classes])
            intermediate_data.tracker_ids = tracker_ids
            '''
            # Update the history after every 3rd recognition.
            if counter % 4 == 0:
                self.updateOutputData()
            if counter > 50:
                counter = 0
            counter += 1
            '''




    class FaceNetModel:
        def __init__(self, SVM_CLASSIFIER_PATH,NN_CLASSIFIER_PATH):
            [self.graph,self.sess] = self.load_model(NN_CLASSIFIER_PATH)
            # Get input and output tensors
            self.images_placeholder = self.graph.get_tensor_by_name("input:0")
            self.embeddings = self.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]
            self.phase_train_placeholder = self.graph.get_tensor_by_name("phase_train:0")
            with open(SVM_CLASSIFIER_PATH, 'rb') as pickle_file:
                self.clf_model, self.clf_class_names = pickle.load(pickle_file)

            # The Image will be resized to this square dimension before computing its vector.
            self.IMAGE_SIZE_RECOGNITION = 160
            print("FaceNetModel. Resize Image to " + str(self.IMAGE_SIZE_RECOGNITION) + 'x' + str(self.IMAGE_SIZE_RECOGNITION))

        def get_representation(self, image_np):
            # Normalize
            image_np = facenet.prewhiten(image_np)
            # Resize the Image
            image_np = cv2.resize(image_np,(self.IMAGE_SIZE_RECOGNITION,self.IMAGE_SIZE_RECOGNITION))
            # BGR -> RGB
            image_np = image_np[:,:,::-1]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Get the representation.
            feed_dict = { self.images_placeholder:image_np_expanded, self.phase_train_placeholder:False }
            return self.sess.run(self.embeddings, feed_dict=feed_dict)

        def get_classification(self, image_np):
            '''
            Returns the index of the face detected and their probabilities.
            '''
            emb_array = self.get_representation(image_np)
            predictions = self.clf_model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            return [best_class_indices,best_class_probabilities]

        def load_model(self,PATH_TO_MODEL):
            GPU_ALLOCATION = 0.2
            print('Loading Model File: ' + PATH_TO_MODEL)
            # Load the graphs.
            [graph, sess] = nn_model_utils.load_graph_with_sess(PATH_TO_MODEL,GPU_ALLOCATION)
            print('Finished Loading Model. Allocated ' + str(GPU_ALLOCATION*100) + "% of GPU."  )
            return [graph, sess]
