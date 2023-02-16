import json
import sys
import time
from networktables import NetworkTables
from networktables import NetworkTablesInstance
import cv2
import MultiThreadedDisplay as MTD
import threading

cscoreAvailable = True
try:
    from cscore import CameraServer
except ImportError:
    cscoreAvailable = False

CAMERA_FPS = 25
DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....


class FRC:
  
    ROMI_FILE = "/boot/romi.json"
    FRC_FILE = "/boot/frc.json"
    NN_FILE = "/boot/nn.json"


    def __init__(self, previewWidth, previewHeight):
        self.team = 0
        self.server = False
        self.hasDisplay = False
        self.ntinst = None
        self.sd = None
        self.frame_counter = 0
        self.LaserDotProjectorCurrent = 0
        self.lastTime = 0

        self.mtd = MTD.MTD(previewWidth, previewHeight)
        self.lockNT = threading.Lock()

        self.gripperImage = None
        self.detectionsImage = None

        self.read_frc_config()

        self.ntinst = NetworkTablesInstance.getDefault()

        if self.server:
            print("Setting up NetworkTables server")
            self.ntinst.startServer()
        else:
            print("Setting up NetworkTables client for team {}".format(self.team))
            self.ntinst.startClientTeam(self.team)
            self.ntinst.startDSClient()

        self.sd = NetworkTables.getTable("MonsterVision")

    

    # Return True if we're running on Romi.  False if we're a coprocessor on a big 'bot

    def is_romi(self):
        try:
            with open(self.ROMI_FILE, "rt", encoding="utf-8") as f:
                json.load(f)
                # j = json.load(f)
        except OSError as err:
            print("Could not open '{}': {}".format(self.ROMI_FILE, err), file=sys.stderr)
            return False
        return True


    def is_frc(self):
        try:
            with open(self.FRC_FILE, "rt", encoding="utf-8") as f:
                json.load(f)
        except OSError as err:
            print("Could not open '{}': {}".format(self.FRC_FILE, err), file=sys.stderr)
            return False
        return True

    def parse_error(self, mess):
        """Report parse error."""
        print("config error in '" + self.FRC_FILE + "': " + mess, file=sys.stderr)


    def read_frc_config(self):

        try:
            with open(self.FRC_FILE, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(self.FRC_FILE, err), file=sys.stderr)
            return False

        # top level must be an object
        if not isinstance(j, dict):
            self.parse_error("must be JSON object")
            return False

        # Is there an desktop display?
        try:
            self.hasDisplay = j["hasDisplay"]
        except KeyError:
            self.hasDisplay = False

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parse_error("could not read team number")
            return False

        # ntmode (optional)
        if "ntmode" in j:
            s = j["ntmode"]
            if s.lower() == "client":
                self.server = False
            elif s.lower() == "server":
                self.server = True
            else:
                self.parse_error("could not understand ntmode value '{}'".format(s))

        # LaserDotProjectorCurrent
        try:
            self.LaserDotProjectorCurrent = j["LaserDotProjectorCurrent"]
        except KeyError:
            self.LaserDotProjectorCurrent = 0

        self.LaserDotProjectorCurrent *= 1.0
        
        return True
    
    
    def writeObjectsToNetworkTable(self, jsonObjects, cam):
        # Protect NT access with a lock.  Just in case NT implementation is not thread-safe
        self.lockNT.acquire()
        self.sd.putString("ObjectTracker-" + cam, jsonObjects)
        self.ntinst.flush()
        self.lockNT.release()

    def displayResults(self, fullFrame, depthFrameColor, detectionFrame, cam):
        if self.hasDisplay:
            now = time.monotonic_ns()
            fps = 1000000000.0 / (now - self.lastTime)
            self.lastTime = now
            cv2.putText(fullFrame, "NN fps: {:.2f}".format(fps), (2, fullFrame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
            (255, 255, 255))

            if depthFrameColor is not None:
                self.mtd.enqueueImage("depth " + cam, depthFrameColor)
                # cv2.imshow("depth " + cam, depthFrameColor)
            if detectionFrame is not None:
                self.mtd.enqueueImage("detections " + cam, detectionFrame)
                # cv2.imshow("detections " + cam, detectionFrame)
            if fullFrame is not None:
                self.mtd.enqueueImage("openCV " + cam, fullFrame)
                # cv2.imshow("openCV " + cam, fullFrame)

        if cam == "Gripper":
            self.gripperImage = fullFrame
        else:
            self.detectionsImage = detectionFrame
        if self.frame_counter % (CAMERA_FPS / DESIRED_FPS) == 0:
            if self.gripperImage is not None and self.detectionsImage is not None:
                img = cv2.vconcat([self.gripperImage, self.detectionsImage])
            elif self.gripperImage is not None:
                img = self.gripperImage
            else:
                img = self.detectionsImage
            # uncomment to turn on local display for HP laptop
            # self.mtd.enqueueImage("DS View", img)
            self.mtd.enqueueImage("DS Image", img)              # Special window name causes MTD to send to camera server
            

        self.frame_counter += 1

        # if self.hasDisplay and cv2.waitKey(1) == ord('q'):
        #     return False
        if self.mtd.areWeDone():
            return False
        
        return True

    def runDisplay(self):
        self.mtd.displayLoop()
        return
