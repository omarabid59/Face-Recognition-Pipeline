import cv2
import numpy as np
class draw_face_frame():
    def __init__(self):
        

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

    def drawDynamicBoundingBoxes(self,image_np, results_data,show_scores=False):
        """
        Identical to 'drawFaceBoundingBoxesAndLabel', except draws custom bouding boxes.
        """

        h,w,_ = image_np.shape
        bbs = results_data.bbs;
        scores = results_data.scores;
        persons = results_data.persons;


        for bb, score,person in zip(bbs,scores,persons):
            person_label = person.split(':')[1]
            if show_scores:
                person_label = person_label + ' ' + str(int(score*100)) + '%'

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


            image_np = self.gen_dynamic_image_label(person_label,x,y,image_np)
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
