import json
import sys
from networktables import NetworkTables
from networktables import NetworkTablesInstance
import cv2

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

    team = 0
    server = False
    hasDisplay = False
    ntinst = None
    sd = None
    frame_counter = 0
    LaserDotProjectorCurrent = 0

    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    

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
    
    def start(self, previewWidth, previewHeight):
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

        if cscoreAvailable:
            self.cs = CameraServer.getInstance()
            self.cs.enableLogging()
            self.output = self.cs.putVideo("MonsterVision", previewWidth, previewHeight) # TODOnot        

    
    
    def writeObjectsToNetworkTable(self, jsonObjects):
        self.sd.putString("ObjectTracker", jsonObjects)
        self.ntinst.flush()

    def displayResults(self, fullFrame, depthFrameColor, detectionFrame):
        if self.hasDisplay:
            if depthFrameColor is not None:
                cv2.imshow("depth", depthFrameColor)
            if detectionFrame is not None:
                cv2.imshow("detections", detectionFrame)
            if fullFrame is not None:
                cv2.imshow("tags", fullFrame)

        if cscoreAvailable:
            if self.frame_counter % (CAMERA_FPS / DESIRED_FPS) == 0:
                self.output.putFrame(detectionFrame)

            self.frame_counter += 1

        if self.hasDisplay and cv2.waitKey(1) == ord('q'):
            return False
        
        return True

