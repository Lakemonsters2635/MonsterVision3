import cv2
import queue

cscoreAvailable = True
try:
    from cscore import CameraServer
except ImportError:
    cscoreAvailable = False

class MTD:
    def __init__(self, previewWidth, previewHeight):
        self.Q = queue.Queue()
        self.allDone = False
        
        if cscoreAvailable:
            # self.cs = CameraServer.getInstance()
            CameraServer.enableLogging()
            self.csoutput = CameraServer.putVideo("MonsterVision", previewWidth, previewHeight) # TODOnot        


    def enqueueImage(self, window : str, image):
        self.Q.put((window, image))

    def displayLoop(self):
        while True:
            (window, image) = self.Q.get(True)
            if window == "DS Image":
                if cscoreAvailable:
                    self.csoutput.putFrame(image)
                continue
            cv2.imshow(window, image)
            self.Q.task_done()
            wk = cv2.waitKey(1)
            self.allDone = self.allDone or wk == 113
            if self.allDone: 
                print("Done")
                return
        return

    def areWeDone(self):
        return self.allDone

