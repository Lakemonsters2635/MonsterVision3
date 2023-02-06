import json
import math
from pathlib import Path
import sys
import time
import depthai as dai
import cv2

INCHES_PER_MILLIMETER = 39.37 / 1000

# Weights to use when blending depth/rgb image (should equal 1.0)
rgbWeight = 0.4
depthWeight = 0.6


def updateBlendWeights(percent_rgb):
    """
    Update the rgb and depth weights used to blend depth/rgb image

    @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
    """
    global depthWeight
    global rgbWeight
    rgbWeight = float(percent_rgb)/100.0
    depthWeight = 1.0 - rgbWeight



def _average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = (pt2[0] - pt1[0]) * factor / 2
    y_shift = (pt2[1] - pt1[1]) * factor / 2
    av_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    av_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return av_pt1, av_pt2


class OAK:
    rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    rgbWidth = 1920
    rgbHeight = 1080
    ispScale = (1, 6)

    bbfraction = 0.2

    CAMERA_FPS = 25
    DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....
    PREVIEW_WIDTH = 200
    PREVIEW_HEIGHT = 200

    syncNN = True

    def __init__(self, laserProjectorNotUsed=None):
        return


    NN_FILE = "/boot/nn.json"

    openvinoVersions = dai.OpenVINO.getVersions()
    openvinoVersionMap = {}
    for v in openvinoVersions:
        openvinoVersionMap[dai.OpenVINO.getVersionName(v)] = v

    def parse_error(self, mess):
        """Report parse error."""
        print("config error in '" + self.NN_FILE + "': " + mess, file=sys.stderr)

    def read_nn_config(self):
        try:
            with open(self.NN_FILE, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(self.NN_FILE, err), file=sys.stderr)
            return {}

        # top level must be an object
        if not isinstance(j, dict):
            self.parse_error("must be JSON object")
            return {}

        return j
    
    def setupSDN(self, nnConfig):

        nnJSON = self.read_nn_config()
        self.LABELS = nnJSON['mappings']['labels']
        nnConfig = nnJSON['nn_config']
    
        # Get path to blob

        blob = nnConfig['blob']
        nnBlobPath = str((Path(__file__).parent / Path('models/' + blob)).resolve().absolute())

        if not Path(nnBlobPath).exists():
            import sys

            raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

        # MobilenetSSD label texts
        self.labelMap = self.LABELS

        # Create pipeline
        self.pipeline = dai.Pipeline()

        try:
            self.openvinoVersion = nnConfig['openvino_version']
        except KeyError:
            self.openvinoVersion = ''

        if self.openvinoVersion != '':
            self.pipeline.setOpenVINOVersion(self.openvinoVersionMap[self.openvinoVersion])

        try:
            self.inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))
        except KeyError:
            self.inputSize = (300, 300)

        family = nnConfig['NN_family']
        if family == 'mobilenet':
            detectionNodeType = dai.node.MobileNetSpatialDetectionNetwork
        elif family == 'YOLO':
            detectionNodeType = dai.node.YoloSpatialDetectionNetwork
        else:
            raise Exception(f'Unknown NN_family: {family}')

        try:
            self.bbfraction = nnConfig['bb_fraction']
        except KeyError:
            self.bbfraction = self.bbfraction			# No change fromn default



        # Create the spatial detection network node - either MobileNet or YOLO (from above)

        spatialDetectionNetwork = self.pipeline.create(detectionNodeType)

        # Set the NN-specific stuff

        if family == 'YOLO':
            spatialDetectionNetwork.setNumClasses(nnConfig['NN_specific_metadata']['classes'])
            spatialDetectionNetwork.setCoordinateSize(nnConfig['NN_specific_metadata']['coordinates'])
            spatialDetectionNetwork.setAnchors(nnConfig['NN_specific_metadata']['anchors'])
            spatialDetectionNetwork.setAnchorMasks(nnConfig['NN_specific_metadata']['anchor_masks'])
            spatialDetectionNetwork.setIouThreshold(nnConfig['NN_specific_metadata']['iou_threshold'])
            spatialDetectionNetwork.setConfidenceThreshold(nnConfig['NN_specific_metadata']['confidence_threshold'])
        else:
            x = nnConfig['confidence_threshold']
            spatialDetectionNetwork.setConfidenceThreshold(x)
        
        spatialDetectionNetwork.setBlobPath(nnBlobPath)
        spatialDetectionNetwork.setConfidenceThreshold(0.5)
        spatialDetectionNetwork.input.setBlocking(False)
        spatialDetectionNetwork.setBoundingBoxScaleFactor(self.bbfraction)
        spatialDetectionNetwork.setDepthLowerThreshold(100)
        spatialDetectionNetwork.setDepthUpperThreshold(5000)

        return spatialDetectionNetwork

    def buildDebugPipeline(self):
        return
        

    def displayDebug(self, device):
        return

    def buildPipeline(self, spatialDetectionNetwork):

        # Define sources and outputs

        self.camRgb = self.pipeline.create(dai.node.ColorCamera)

        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb.setStreamName("rgb")

        # Properties

        # self.camRgb.setPreviewSize(self.inputSize)
        self.camRgb.setResolution(self.rgbResolution)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(self.CAMERA_FPS)
        self.camRgb.setIspScale(self.ispScale[0], self.ispScale[1])

        print("Camera FPS: {}".format(self.camRgb.getFps()))

        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        
        try:
            calibData = dai.Device().readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except Exception:
           pass 

        # Linking

        self.camRgb.isp.link(self.xoutRgb.input)

        # Extra for debugging

        self.buildDebugPipeline()



    def runPipeline(self, processDetections, objectsCallback=None, displayResults=None, processImages=None, cam="", imagesParam=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            self.fps = 0

            while True:
                self.inRgb = rgbQueue.get()
                self.frame = self.inRgb.getCvFrame()

                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    self.fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                objects = []

                if processImages is not None:
                    additionalObjects = processImages(self.frame, None, None, imagesParam)
                    if additionalObjects is not None:
                        objects = objects + additionalObjects

                if objectsCallback is not None:
                    objectsCallback(objects, cam)

                self.displayDebug(device)

                if displayResults is not None:
                    if displayResults(self.frame, None, None, cam) == False:
                        return


    def processDetections(self, detections):
        return
                