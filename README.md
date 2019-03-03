# Face-Recognition-Pipeline
A robust face recognition pipeline using FaceNet as a backend. A moving average algorithm is used to give a better prediction of the person being detected.


## Quick Start Guide
Grab the code from this GitHub repository.  

```git clone --recurse-submodules git@github.com:omarabid59/Face-Recognition-Pipeline.git```

Import the FaceDTCM package  
```
import FaceDTCM.utils as dtcmUtils
import FaceDTCM.Pipeline as faceDtcmPipeline
```

Initialize the web camera so we can get video input to feed into our pipeline. Initializing the camera returns a pointer to a thread which holds the webcam data.  
```thread_img = dtcmUtils.init_camera()```

Specify the path to our Neural Network and SVM classifier.  
```
SVM_CLASSIFIER = "/path/to/svm"
NN_CLASSIFIER_PATH = "/path/to/nn/classifier"
```
Initialize the Face Detection, Tracking, Recognition and Memory pipeline  
```
face_pipeline = faceDtcmPipeline(thread_img.image_data,
                        SVM_CLASSIFIER,
                        NN_CLASSIFIER_PATH,
                        detector_thresh = 0.5,
                        recognition_thresh=0.5)
```

### Run the Pipeline
The simplest way to get the pipeline running is to start all processes. We can do this with the ``startAll()`` function.
```
face_pipeline.startAll()
```

### Visualize the results
```
faceFrame = draw_face_frame()
while True:
    # Get the webcam image
    live_img = thread_img.image_data.image_np.copy()
    data = face_pipeline.detect_and_recognize()
    #PERSONS = list(wrapper.output_data.output_data.recognition_data.persons)
    persons = data.persons

    person_img = faceFrame.drawFaceRecognitionDisplayImage(PERSONS)
    live_img = faceFrame.drawDynamicBoundingBoxes(live_img, thread_frm.output_data.output_data)


    cv2.destroyAllWindows()
```
