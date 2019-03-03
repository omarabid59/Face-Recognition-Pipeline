# Add the utlities to the path
import sys
import numpy as np
from ImageInput import WebcamThread
from Predictor.Pipeline import FaceDetectionRecognition as FaceDTCM
import matplotlib.pyplot as plt


from gui_functions import draw_face_frame
import cv2
print("Finished Loading Imports")


# Parameters
detector_thresh = 0.8
recognition_thresh = 0.5
TRACKER = 'Legacy'
RESOLUTIONS = [[640,480],[1280,720],[1920,1080]]
FRAME_SCALE = 1.0
SVM_CLASSIFIER = '/home/watopedia/github_projects/aipod-data/face_recognition/train_datasets/watopedia-2018-10-13/svm_models/facenet_mtcnn_align/classifier.pkl'
NN_CLASSIFIER_PATH = 'UPDATE_PATH'

def init_camera(resolution):
    thread_image = WebcamThread('camera',VIDEO_ID,
                                IMAGE_WIDTH=resolution[0],
                                       IMAGE_HEIGHT=resolution[1])
    image_data = thread_image.image_data
    thread_image.start()
    while len(thread_image.image_data.image_np) == 0:
        time.sleep(0.1)
    print('Camera Initialized')
    return thread_image,image_data
def init_pipeline(thread_image):
    wrapper = FaceDTCM(thread_image.image_data,
                        SVM_CLASSIFIER,
                        NN_CLASSIFIER_PATH,
                        detector_thresh = detector_thresh,
                        recognition_thresh=recognition_thresh)
    return wrapper;

def run_live():
    thread_image,image_data = init_camera(RESOLUTIONS[2])
    wrapper = init_pipeline(thread_image)
    faceFrame = draw_face_frame()
    while True:
        # Get the webcam image
        live_img = thread_image.image_data.image_np.copy()
        # Concatenate the output data from all of our threads.
        output_data = wrapper.output_data
        PERSONS = list(wrapper.output_data.output_data.recognition_data.persons)

        person_img = faceFrame.drawFaceRecognitionDisplayImage(PERSONS)
        live_img = faceFrame.drawDynamicBoundingBoxes(live_img, thread_frm.output_data.output_data)


        cv2.destroyAllWindows()
def train_classifier():
    '''
    TODO: Implementation required. Should take in a set of images and train a classifier.
    Alternatively, use a pre-existing SVM classifier.
    '''
    pass
