# Add the utlities to the path
import sys
import numpy as np
from ImageInput import WebcamThread
from Predictor.Wrapper import FaceDetectionRecognition as FaceNetRecog
import matplotlib.pyplot as plt


from gui_functions import draw_face_frame
import cv2
print("Finished Loading Imports")


# Parameters
detector_thresh = 0.8
recognition_thresh = 0.5
DETECTOR_IMG_SCALE = 0.5
TRACKER = 'Legacy'
RESOLUTIONS = [[640,480],[1280,720],[1920,1080]]
FRAME_SCALE = 1.0

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
def init_pipeline():
    thread_frm = FaceNetRecog(thread_image.image_data,
                      '/home/watopedia/github_projects/aipod-data/face_recognition/train_datasets/watopedia-2018-10-13/svm_models/facenet_mtcnn_align/classifier.pkl',
                                 detector_thresh,
                                 recognition_thresh=recognition_thresh,
                                 detection_tracker=TRACKER,
                                 DETECTOR_IMG_SCALE = DETECTOR_IMG_SCALE)
    return thread_frm;






def run():
    thread_image,image_data = init_camera(RESOLUTIONS[2])
    thread_frm = init_pipeline()
    faceFrame = draw_face_frame()
    while True:
        # Get the webcam image
        live_img = thread_image.image_data.image_np.copy()
        # Concatenate the output data from all of our threads.
        output_data = thread_frm.output_data


        PERSONS = list(thread_frm.output_data.output_data.recognition_data.persons)

        person_img = faceFrame.drawFaceRecognitionDisplayImage(PERSONS)
        live_img = faceFrame.drawDynamicBoundingBoxes(live_img, thread_frm.output_data.output_data)


        cv2.destroyAllWindows()
