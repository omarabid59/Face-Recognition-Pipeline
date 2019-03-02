# Root Directory
import os.path
ROOT_DIR = '/' + os.path.join(*__file__.split('/')[:-2]) + '/'


# Setup Face Recognition Constants
frm_base_dir = ROOT_DIR + 'nnModels/faceRecognition/'
FACE_SHAPE_PREDICTOR = frm_base_dir + 'shape_predictor_68_face_landmarks.dat'
FACE_SHAPE_DNN_MODEL = frm_base_dir + 'nn4.small2.v1.t7'
FACE_SHAPE_DNN_MODEL_2 = frm_base_dir + '20180408-102900.pb'
FACE_CLASSIFIER_MODEL = frm_base_dir + 'repClassifiers/openface-wo-alignment.pkl'
FACE_DETECTOR_SSD_MODEL = frm_base_dir + 'ssd_face_detector/ssd_inference_graph.pb'
FACE_DETECTOR_SSD_MODEL_LABEL = frm_base_dir + 'ssd_face_detector/face_label_map.pbtxt'
FACENET_DNN_MODEL = frm_base_dir + 'faceNet/20180408-102900/20180408-102900.pb'
FACENET_CLASSIFIER = frm_base_dir + 'faceNet/20180408-102900/20181018-watopedia.pkl'
