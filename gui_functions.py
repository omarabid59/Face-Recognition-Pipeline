import aipodMain.GUI.gui_loader_utils as gui_loader_utils
import constants
import cv2
import numpy as np
class draw_face_frame():
    def __init__(self):
        self.FACE_DB_IMAGES_DIR = constants.FACE_DB_IMAGES_DIR
        self.HEIGHT = constants.FRAME_FACERECOG_HEIGHT
        self.WIDTH = constants.FRAME_FACERECOG_WIDTH

        self.__loadModelFiles()

        self.__font      = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1.0
        self.__fontColor = (0,0,0)
        self.__lineType  = 2

        self.rectColor = (255,0,0)


        # VARIABLES FOR THE DYNAMIC BOUDNING BOX DRAWING
        self.__LINE_COLOR = (188,42,22)
        self.__LINE_THICKNESS = 5
        self.__ALPHA_MIN = 0.2
        self.__ALPHA_MAX = 0.9
        self.__ALPHA_VALUES = list(np.linspace(self.__ALPHA_MIN,self.__ALPHA_MAX,num=10)) + list(np.linspace(self.__ALPHA_MAX,self.__ALPHA_MIN,num=7))
        self.__DYNAMIC_BOX_COUNTER = 0

    def __loadModelFiles(self):
        # Load the Face Identification
        self.face_id_dict = gui_loader_utils.load_face_model_images(self.FACE_DB_IMAGES_DIR,
                                                               self.WIDTH,
                                                               self.HEIGHT)
        print('Loaded 3D models: ' + str(self.face_id_dict.keys()))


    def __updateFaceRecognitionDisplayImage(self, persons):
        '''
        Passes an array of strings and the face image database to return
        the corresponding image of the person being detected.
        Used to display image on the dashboard
        '''
        if len(persons) > 0:
            if 'Alex' in persons[0]:
                 current_face_model = self.face_id_dict['Alex']
            elif 'Brent-Pass' in persons[0]:
                 current_face_model = self.face_id_dict['Brent']
            elif 'Carl-freer' in persons[0]:
                 current_face_model = self.face_id_dict['Carl']
            elif 'Haris' in persons[0]:
                 current_face_model = self.face_id_dict['Haris']
            elif 'Jan' in persons[0]:
                 current_face_model = self.face_id_dict['Jan']
            elif 'Lucie-parker' in persons[0]:
                 current_face_model = self.face_id_dict['Lucie']
            elif 'omar' in persons[0]:
                 current_face_model = self.face_id_dict['Omar']
            elif 'phanikumar' in persons[0]:
                 current_face_model = self.face_id_dict['Phani']
            else:
                 current_face_model = self.face_id_dict['mask']
        else:
            current_face_model = self.face_id_dict['mask']
        self.current_face_model = current_face_model

    def drawFaceRecognitionDisplayImage(self,person_labels):
        # Check if the label exists.
        indices = [i for i, s in enumerate(person_labels) if 'frm' in s]
        if len(indices) == 1:
            persons = person_labels[indices[0]]
            persons = persons.split(':')[1]
        else:
            persons = '--------'
        self.__updateFaceRecognitionDisplayImage([persons])
        return self.current_face_model


    def drawDynamicBoundingBoxes(self,image_np, output_data,SCORES=False):
        """
        Identical to 'drawFaceBoundingBoxesAndLabel', except draws custom bouding boxes.
        """

        h,w,_ = image_np.shape
        trk_ids = list(output_data.recognition_data.output_data.tracker_ids)
        persons = list(output_data.recognition_data.output_data.classes)
        scores = list(output_data.recognition_data.scores)


        for bb, detector_trk_id in zip(output_data.detection_data.bbs,
                                       output_data.detection_data.tracker_ids):
            try:
                indx = trk_ids.index(detector_trk_id)
                person = persons[indx]
                if SCORES:
                    person = person + ' ' + str(int(scores[indx]*100)) + '%'
            except:
                person = output_data.recognition_data.EMPTY_ELEMENT
            person_text = person.split(':')[1]
            x = int(bb[1]*w)
            y = int(bb[0]*h)
            x_ = int(bb[3]*w)
            y_ = int(bb[2]*h)

            box_width = x_ - x
            box_height = y_ - y
            LINE_LENGTH = int(0.3*min(box_width,box_height))



            overlay = image_np.copy()
            cv2.line(overlay,(x,y),(x + LINE_LENGTH,y),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x_ - LINE_LENGTH,y),(x_,y),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x,y_),(x + LINE_LENGTH,y_),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x_ - LINE_LENGTH,y_),(x_,y_),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x,y),(x,y + LINE_LENGTH),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x_,y),(x_,y + LINE_LENGTH),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x,y_ - LINE_LENGTH),(x,y_),self.__LINE_COLOR,self.__LINE_THICKNESS)
            cv2.line(overlay,(x_,y_ - LINE_LENGTH),(x_,y_),self.__LINE_COLOR,self.__LINE_THICKNESS)

            ALPHA = self.__ALPHA_VALUES[self.__DYNAMIC_BOX_COUNTER]
            cv2.addWeighted(overlay, ALPHA, image_np, 1 - ALPHA,
                0, image_np)
            self.__DYNAMIC_BOX_COUNTER += 1
            if self.__DYNAMIC_BOX_COUNTER > len(self.__ALPHA_VALUES) - 1:
                self.__DYNAMIC_BOX_COUNTER = 0


            image_np = self.gen_dynamic_image_label(person_text,x,y,image_np)
        return image_np

    def gen_dynamic_image_label(self,text,x,y,image_np):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        fontColor = (255,255,255)
        bgColor = (0,0,0)
        fontThicknes = 6
        padding = 40
        fixed_height = int(0.08333333333333333*image_np.shape[0])
        y = int(y - 0.1*image_np.shape[0])
        WIDTH = int(0.2*image_np.shape[1])
        WIDTH_PADDING = int(0.3*WIDTH)
        y_offset = 30
        text_width = WIDTH + 1

        text = text.replace('-',' ')
        while True:
            (text_width,text_height),_ = cv2.getTextSize(text,fontFace,fontScale,fontThicknes)
            if text_width >= (WIDTH - WIDTH_PADDING):
                fontScale -= 0.2
            else:
                break

        image_np = cv2.rectangle(image_np,(x,y),(x+text_width+padding,y+fixed_height),  bgColor,cv2.FILLED)
        image_np = cv2.putText(image_np,text,(x + int(padding)//2,
                                        y + y_offset),
                                    fontFace, fontScale,fontColor,fontThicknes,cv2.LINE_AA)


        return image_np
