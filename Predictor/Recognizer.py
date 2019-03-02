from .AbstractSubsetImgPredictor import AbstractSubsetImgPredictor

import pickle
import threading
import time
import warnings
import facenet
import cv2
import aipodMain.utilities.constants as constants
import aipodMain.utilities.helper_nn_models_utils as nn_model_utils
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
        self.pause = True
        self.name = "Face Recognition Thread."

        self.output_data = face_data


        self.net = self.FaceNetModel(SVM_CLASSIFIER_PATH,NN_CLASSIFIER_PATH)

        self.update_interval_ms = 20



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
                time.sleep(self.update_interval_ms/1000.0)
            else:
                scores = np.asarray([]).reshape(-1,1)
                classes = []
                tracker_ids = []
                time.sleep(1.0)

            intermediate_data.scores = np.asarray(list(scores))
            intermediate_data.classes = list(['frm:' + s for s in classes])
            intermediate_data.tracker_ids = tracker_ids
            # Update the history after every 3rd recognition.
            if counter % 4 == 0:
                self.updateOutputData()
            if counter > 50:
                counter = 0
            counter += 1

    def updateOutputData(self):
        # Get a reference to make it easier to work with
        recognition = self.output_data.recognition_data
        recognition_intermediate = self.output_data.recognition_data.intermediate_data
        history = recognition.history


        person_labels = list(recognition_intermediate.classes)
        tracker_ids = list(recognition_intermediate.tracker_ids)
        scores = list(recognition_intermediate.scores)


        for person, tracker_id, score in zip(person_labels, tracker_ids,scores):
            # Skip if we have an empty string.
            if person.split(':')[1] == '':
                continue
            if tracker_id in history.tracker_ids:
                indx = history.tracker_ids.index(tracker_id)
                history.idle_age[indx] = 0 # Reset the age.
            else:
                # Create a new entry for this face
                history.idle_age.append(0)
                history.recognition_history.append(recognition.EMPTY_MAJORITY_ELEMENTS)
                history.tracker_ids.append(tracker_id)
                history.persons.append(recognition.EMPTY_ELEMENT)
                history.scores.append(0.0)
                indx = len(history.age) - 1

            recognition_history = history.recognition_history[indx]
            recognition_history = self.appendElement(recognition_history,person)
            [recognition_history, majorityElement] = self.majorityVote(recognition_history,
                                                                    person, recognition.EMPTY_ELEMENT)

            history.persons[indx] = majorityElement
            history.scores[indx] = score
            history.recognition_history[indx] = recognition_history


        # These are the trackers that we need to increment the age of.
        unused_trackers = list(set(history.tracker_ids) - set(tracker_ids))
        for tracker_id in unused_trackers:
            indx = history.tracker_ids.index(tracker_id)
            history.idle_age[indx] += 1
            if history.idle_age[indx] > history.MAX_AGE:
                del history.idle_age[indx]
                del history.recognition_history[indx]
                del history.tracker_ids[indx]
                del history.persons[indx]
                del history.scores[indx]
        # This is the output we wish to utilize.
        recognition.output_data.tracker_ids = history.tracker_ids
        recognition.output_data.persons = history.persons
        recognition.output_data.scores = history.scores


    def appendElement(self,recognition_result,new_element):
        """
        Appends the `new_element` to the list of `recognition_result`.
        """
        history_length = self.output_data.recognition_data.history.RECOGNITION_HISTORY_LENGTH
        # Append the new element
        recognition_result.append(new_element)
        # Remove the last result.
        if len(recognition_result) >= history_length:
            recognition_result = recognition_result[-history_length:]
        return recognition_result

    def majorityVote(self,recognition_result, new_element, default):
        """
        Performs the majority vote and returns the result.
        Find which element in *seq* sequence is in the majority.

        Return *default* if no such element exists.

        Use Moore's linear time constant space majority vote algorithm
        """
        majorityElement = default
        count = 0
        for e in recognition_result:
            if count != 0:
                count += 1 if majorityElement == e else -1
            else: # count == 0
                majorityElement = e
                count = 1

        # check the majority
        majorityElement if recognition_result.count(majorityElement) > len(recognition_result) // 2 else default
        return [recognition_result, majorityElement]


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
