import json
import math
from pathlib import Path
import sys
import time
import depthai as dai
import cv2

import notinh

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
    LaserDotProjectorCurrent = 0

    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P
    monoWidth = 1280
    monoHeight = 720

    rgbResolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
    rgbWidth = 1920
    rgbHeight = 1080

    bbfraction = 0.2

    CAMERA_FPS = 25
    DESIRED_FPS = 10		# seem to actually get 1/2 this.  Don't know why.....
    PREVIEW_WIDTH = 200
    PREVIEW_HEIGHT = 200

    syncNN = True

    def __init__(self, LaserDotProjectorCurrent=None):
        self.LaserDotProjectorCurrent = LaserDotProjectorCurrent


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

        # self.manip = self.pipeline.create(dai.node.ImageManip)
        # self.manip.setMaxOutputFrameSize(1382400)
        # self.xoutStereoDepth = self.pipeline.create(dai.node.XLinkOut)
        # self.xoutStereoDepth.setStreamName("stereo-depth")
        # self.xoutIsp = self.pipeline.create(dai.node.XLinkOut)

        # self.ispScale = (2,3)
        # self.camRgb.setIspScale(self.ispScale[0], self.ispScale[1])


        # # splice them into the pipeline

        # if True:
        #     self.manip.out.link(self.xoutIsp.input)
        #     self.camRgb.isp.link(self.manip.inputImage)
        #     self.strName1 = "manip"
        # else:
        # self.camRgb.isp.link(self.xoutIsp.input)
        # self.strName1 = "isp"

        # self.xoutIsp.setStreamName(self.strName1)
        # self.stereo.depth.link(self.xoutStereoDepth.input)

        # cv2.namedWindow("blended")
        # cv2.createTrackbar('RGB Weight %', "blended", int(rgbWeight*100), 100, updateBlendWeights)
        return
        

    def displayDebug(self, device):
        # ispQueue = device.getOutputQueue(name=self.strName1, maxSize=4, blocking=False)
        # isp = ispQueue.get()
        # self.ispFrame = isp.getCvFrame()

        # xScale = self.monoWidth / self.inputSize[0]
        # yScale = self.monoHeight / self.inputSize[1]

        # if xScale > yScale:
        #     dim = (int(self.inputSize[1]*self.monoWidth/self.monoHeight), int(self.inputSize[1]))
        #     xmin = int((dim[0] - self.inputSize[0])/2)
        #     xmax = int(dim[0] - xmin)
        #     ymin = 0
        #     ymax = self.inputSize[1]
        # else:
        #     dim = (int(self.inputSize[0]), int(self.inputSize[0]*self.monoHeight/self.monoWidth))
        #     ymin = int((dim[1] - self.inputSize[1])/2)
        #     ymax = int(dim[1] - ymin)
        #     xmin = 0
        #     xmax = self.inputSize[0]

        # x = dim[0] / self.inputSize[0]
        # y = dim[1] / self.inputSize[1]

        # r = cv2.resize(ispFrame, dim)
        # s = r[ymin:ymax, xmin:xmax]
        # cv2.imshow(self.strName1, self.ispFrame)

        # ispQstereoDepthQueue = device.getOutputQueue(name="stereo-depth", maxSize=4, blocking=False)
        # dpt = ispQstereoDepthQueue.get()
        # dptFrame = dpt.getCvFrame()

        # dfc = cv2.normalize(dptFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # dfc = cv2.equalizeHist(dfc)
        # dfc = cv2.applyColorMap(dfc, cv2.COLORMAP_RAINBOW)

        # # cv2.imshow("stereo-depth", dfc)

        # blended = cv2.addWeighted(ispFrame, rgbWeight, dfc, depthWeight, 0)
        # cv2.imshow("blended", blended)
        return

    def buildPipeline(self, spatialDetectionNetwork):

        # Define sources and outputs

        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
        self.monoRight = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutNN = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutIsp = self.pipeline.create(dai.node.XLinkOut)

        self.xoutRgb.setStreamName("rgb")
        self.xoutNN.setStreamName("detections")
        self.xoutDepth.setStreamName("depth")

        # Properties

        self.camRgb.setPreviewSize(self.inputSize)
        self.camRgb.setResolution(self.rgbResolution)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(self.CAMERA_FPS)
        self.ispScale = (2,3)
        self.camRgb.setIspScale(self.ispScale[0], self.ispScale[1])

        print("Camera FPS: {}".format(self.camRgb.getFps()))

        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        
        try:
            calibData = dai.Device().readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise

        self.monoLeft.setResolution(self.monoResolution)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(self.monoResolution)
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Setting node configs

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        self.stereo.setOutputSize(self.monoLeft.getResolutionWidth(), self.monoLeft.getResolutionHeight())

        # Linking

        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)

        self.camRgb.preview.link(spatialDetectionNetwork.input)
        if self.syncNN:
            spatialDetectionNetwork.passthrough.link(self.xoutRgb.input)
        else:
            self.camRgb.preview.link(self.xoutRgb.input)

        spatialDetectionNetwork.out.link(self.xoutNN.input)

        self.stereo.depth.link(spatialDetectionNetwork.inputDepth)
        spatialDetectionNetwork.passthroughDepth.link(self.xoutDepth.input)

        self.camRgb.isp.link(self.xoutIsp.input)
        self.xoutIsp.setStreamName("isp")

        # Extra for debugging

        self.buildDebugPipeline()



    def runPipeline(self, processDetections, objectsCallback=None, displayResults=None, processImages=None):
        # Connect to device and start pipeline
        with dai.Device(self.pipeline) as device:
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
            try:
                calibData = device.readCalibration2()
                lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
                if lensPosition:
                    self.camRgb.initialControl.setManualFocus(lensPosition)
            except:
                raise

            # Output queues will be used to get the rgb frames and nn data from the outputs defined above
            previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
            depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            ispQueue = device.getOutputQueue(name="isp", maxSize=4, blocking=False)

            startTime = time.monotonic()
            counter = 0
            self.fps = 0

            if self.LaserDotProjectorCurrent is not None:
                device.setIrLaserDotProjectorBrightness(self.LaserDotProjectorCurrent)

            while True:
                self.inPreview = previewQueue.get()
                self.inDet = detectionNNQueue.get()
                self.depth = depthQueue.get()
                self.isp = ispQueue.get()

                counter += 1
                current_time = time.monotonic()
                if (current_time - startTime) > 1:
                    self.fps = counter / (current_time - startTime)
                    counter = 0
                    startTime = current_time

                self.frame = self.inPreview.getCvFrame()
                self.depthFrame = self.depth.getFrame()
                self.ispFrame = self.isp.getCvFrame()

                self.depthFrameColor = cv2.normalize(self.depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                self.depthFrameColor = cv2.equalizeHist(self.depthFrameColor)
                self.depthFrameColor = cv2.applyColorMap(self.depthFrameColor, cv2.COLORMAP_RAINBOW)

                detections = self.inDet.detections
                if len(detections) != 0:
                    objects = processDetections(self, detections)
                    if objects is None:
                        objects = []
                else:
                    objects = []

                if processImages is not None:
                    additionalObjects = processImages(self.ispFrame, self.depthFrame, self.depthFrameColor)
                    if additionalObjects is not None:
                        objects = objects + additionalObjects

                if objectsCallback is not None:
                    objectsCallback(objects)

                self.displayDebug(device)

                if displayResults is not None:
                    if displayResults(self.ispFrame, self.depthFrameColor, self.frame) == False:
                        return


    def processDetections(self, detections):

        # If the frame is available, draw bounding boxes on it and show the frame
        height = self.frame.shape[0]
        width = self.frame.shape[1]

        # re-initializes objects to zero/empty before each frame is read
        objects = []
        s_detections = sorted(detections, key=lambda det: det.label * 100000 + det.spatialCoordinates.z)

        for detection in s_detections:
            roi = detection.boundingBoxMapping.roi
            roi = roi.denormalize(self.depthFrameColor.shape[1], self.depthFrameColor.shape[0])
            topLeft = roi.topLeft()
            bottomRight = roi.bottomRight()
            xmin = int(topLeft.x)
            ymin = int(topLeft.y)
            xmax = int(bottomRight.x)
            ymax = int(bottomRight.y)

            cv2.rectangle(self.depthFrameColor, (xmin, ymin), (xmax, ymax), 255, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

            # Denormalize bounding box

            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)

            try:
                label = self.labelMap[detection.label]

            except KeyError:
                label = detection.label

            # Draw the BB over which the depth is computed
            avg_pt1, avg_pt2 = _average_depth_coord([detection.xmin, detection.ymin],
                                                   [detection.xmax, detection.ymax],
                                                   self.bbfraction)
            avg_pt1 = int(avg_pt1[0] * width), int(avg_pt1[1] * height)
            avg_pt2 = int(avg_pt2[0] * width), int(avg_pt2[1] * height)

            cv2.rectangle(self.frame, avg_pt1, avg_pt2, (0, 255, 255), 1)
            # Choose the color based on the label

            if detection.label == 1:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)

            #print(detection.spatialCoordinates.x, detection.spatialCoordinates.y, detection.spatialCoordinates.z)

            x = round(int(detection.spatialCoordinates.x * INCHES_PER_MILLIMETER), 1)
            y = round(int(detection.spatialCoordinates.y * INCHES_PER_MILLIMETER), 1)
            z = round(int(detection.spatialCoordinates.z * INCHES_PER_MILLIMETER), 1)

            cv2.putText(self.frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, "{:.2f}".format(detection.confidence * 100), (x1 + 10, y1 + 35),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"X: {x} in", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"Y: {y} in", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(self.frame, f"Z: {z} in", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

            objects.append({"objectLabel": self.LABELS[detection.label], "x": x,
                            "y": y, "z": z,
                            "confidence": round(detection.confidence, 2)})

        cv2.putText(self.frame, "NN fps: {:.2f}".format(self.fps), (2, self.frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4,
                    (255, 255, 255))

                