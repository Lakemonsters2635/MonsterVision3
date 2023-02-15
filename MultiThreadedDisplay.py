import cv2
import queue


class MTD:
    def __init__(self):
        self.Q = queue.Queue()
        self.allDone = False

    def enqueueImage(self, window : str, image):
        self.Q.put((window, image))

    def displayLoop(self):
        while True:
            (window, image) = self.Q.get(True)
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

