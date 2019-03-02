import cv2
import re
import subprocess
import os
import threading
import time
from DataHolder import ImageData
class WebcamThread(threading.Thread):
    def __init__(self,VIDEO_ID, IMAGE_WIDTH = 640 ,IMAGE_HEIGHT = 480):
        threading.Thread.__init__(self)
        ABC.__init__(self)
        self.name = "Webcam Thread"
        self.image_data = ImageData()
        self.done = False
        self.IMAGE_WIDTH = IMAGE_WIDTH
        self.IMAGE_HEIGHT = IMAGE_HEIGHT
        self.cap = []
        self.cap = self.init_input(IMAGE_WIDTH,IMAGE_HEIGHT, VIDEO_ID)
        self.frame_per_ms = 50.0

    def init_input(self, IMAGE_WIDTH,IMAGE_HEIGHT, VIDEO_ID):
        cap = cv2.VideoCapture(VIDEO_ID)
        assert cap.isOpened() == True, 'Could not open Webcam.'
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)
        return cap

    def getImage(self):
        return self.image_data.image_np

    def initParams(self,image_np):
        h,w,_ = image_np.shape
        self.image_data.height = h
        self.image_data.width = w
        self.image_data.isInit = True

    def updateImg(self, threadName):
        self.initParams(self.cap.read()[1])
        while not self.done:
            _, self.image_data.image_np = self.cap.read()
            time.sleep(self.frame_per_ms/1000.0)

    def run(self):
        print("Starting " + self.name)
        self.updateImg(self.name)
        print("Exiting " + self.name)
    def stop(self):
        self.done = True
        self.cap.release()
