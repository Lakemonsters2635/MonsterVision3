import cv2
import queue
import threading


# NOTE: This class should only be used as a singleton

class MTD:
    Q = queue.Queue()
    allDone = False

    def enqueueImage(self, window : str, image):
        self.Q.put((window, image))
        # print(f"{image[0][0]} => {window}")

    def displayLoop(self):
        while True:
            (window, image) = self.Q.get(True)
            # print(f"{image[0][0]} <= {window}")
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


    # def __init__(self):
    #     thread = threading.Thread(target=displayLoopStatic, args=(self, ))
    #     thread.start()
