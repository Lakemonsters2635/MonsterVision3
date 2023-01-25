import json
import math
from pathlib import Path
import sys
import time
import depthai as dai
import cv2
import numpy as np

INCHES_PER_MILLIMETER = 39.37 / 1000

def _average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = (pt2[0] - pt1[0]) * factor / 2
    y_shift = (pt2[1] - pt1[1]) * factor / 2
    av_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    av_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return av_pt1, av_pt2



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



class OAK:
    LaserDotProjectorCurrent = 0

    monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

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
        # return None

    def buildDebugPipeline(self):
        # scaleIt = 1.17

        # htScale = self.rgbHeight / self.inputSize[1]
        # wdScale = self.rgbWidth / self.inputSize[0]

        # if htScale <= wdScale:
        #     gcd = math.gcd(self.rgbHeight, self.inputSize[1])
        #     self.ispScale = (int(self.inputSize[1] / gcd), int(self.rgbHeight / gcd))

        #     scale = self.ispScale[0] / self.ispScale[1]
        #     ispWidth = self.rgbWidth * scale
        #     ispHeight = self.rgbHeight * scale

        #     xCrop = 1.0 - self.inputSize[0] / ispWidth
        #     yCrop = 1.0 - self.inputSize[1] / ispHeight

        #     self.xCropMin = xCrop/2
        #     self.xCropMax = 1.0 - xCrop/2
        #     self.yCropMin = yCrop/2
        #     self.yCropMax = 1.0 - yCrop/2
        # else:
        #     gcd = math.gcd(self.rgbWidth, self.inputSize[0])
        #     self.ispScale = (int(self.inputSize[0] / gcd), int(self.rgbWidth / gcd))
        #     self.xCropMin = 0.0
        #     self.xCropMax = 1.0
        #     yCrop = 1.0 - (self.inputSize[1] * wdScale / self.rgbHeight)
        #     self.yCropMin = yCrop/2
        #     self.yCropMax = 1.0 - yCrop/2

        # self.ispScale = (1,1)
        # self.ispWidth = self.rgbWidth * self.ispScale[0]/self.ispScale[1]
        # self.ispHeight = self.rgbHeight * self.ispScale[0]/self.ispScale[1]

        # # define additional sources and outputs

        # self.xoutIsp = self.pipeline.create(dai.node.XLinkOut)
        # self.manip = self.pipeline.create(dai.node.ImageManip)
        # self.xoutStereoDepth = self.pipeline.create(dai.node.XLinkOut)
        # self.xoutIsp.setStreamName("isp")
        # self.xoutStereoDepth.setStreamName("stereo-depth")

        # self.manip.setMaxOutputFrameSize(3110400)
        # # self.manip.initialConfig.setCropRect(self.xCropMin, self.yCropMin, self.xCropMax, self.yCropMax)
        # # self.manip.initialConfig.setResize(0.25)
        # cc = self.manip.initialConfig.getCropConfig()
        # xmin = self.manip.initialConfig.getCropXMin()
        # xmax = self.manip.initialConfig.getCropXMax()
        # ymin = self.manip.initialConfig.getCropYMin()
        # ymax = self.manip.initialConfig.getCropYMax()

        # ht = self.manip.initialConfig.getResizeHeight()
        # wd = self.manip.initialConfig.getResizeWidth()

        # self.camRgb.setIspScale(self.ispScale[0], self.ispScale[1])


        # # splice them into the pipeline

        # if False:
        #     self.manip.out.link(self.xoutIsp.input)
        #     self.camRgb.isp.link(self.manip.inputImage)
        # else:
        #     self.camRgb.isp.link(self.xoutIsp.input)

        # self.stereo.depth.link(self.xoutStereoDepth.input)
        return

    def displayDebug(self, device):
        # ispQueue = device.getOutputQueue(name="isp", maxSize=4, blocking=False)
        # isp = ispQueue.get()
        # ispFrame = isp.getCvFrame()

        # scale = (8.0 / 27.0)# * 1.2
        # scale = (8.0 / 27.0) 

        # scaledHeight = ispFrame.shape[0] * scale
        # scaledWidth = ispFrame.shape[1] * scale
        # cropW = self.inputSize[0] / scaledWidth
        # cropH = self.inputSize[1] / scaledHeight

        # dim = (int(scaledWidth), int(scaledHeight))
        # xmin = int((scaledWidth - self.inputSize[0]) / 2)
        # xmax = int(xmin + self.inputSize[0])
        # ymin = int((scaledHeight - self.inputSize[1]) / 2)
        # ymax = int(ymin + self.inputSize[1])

        # cv2.imshow("isp", cv2.resize(ispFrame, dim)[ymin:ymax, xmin:xmax])

        # ispQstereoDepthQueue = device.getOutputQueue(name="stereo-depth", maxSize=4, blocking=False)
        # dpt = ispQstereoDepthQueue.get()
        # dptFrame = dpt.getCvFrame()

        # dfc = cv2.normalize(dptFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # dfc = cv2.equalizeHist(dfc)
        # dfc = cv2.applyColorMap(dfc, cv2.COLORMAP_RAINBOW)

        # cv2.imshow("stereo-depth", dfc)
        return None

    def buildPipeline(self, spatialDetectionNetwork):
        # Optional. If set (True), the ColorCamera is downscaled from 1080p to 720p.
        # Otherwise (False), the aligned depth is automatically upscaled to 1080p
        self.downscaleColor = True
        self.fps = 30
        # The disparity is computed at this resolution, then upscaled to RGB resolution
        monoResolution = dai.MonoCameraProperties.SensorResolution.THE_720_P

        # Create pipeline
        # self.pipeline = dai.Pipeline()                                                # **Done in SDN**
        self.device = dai.Device()
        self.queueNames = []

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.left = self.pipeline.create(dai.node.MonoCamera)
        self.right = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        self.rgbOut = self.pipeline.create(dai.node.XLinkOut)
        self.disparityOut = self.pipeline.create(dai.node.XLinkOut)
        self.leftOut = self.pipeline.create(dai.node.XLinkOut)
        self.NNout = self.pipeline.create(dai.node.XLinkOut)

        self.rgbOut.setStreamName("rgb")
        self.queueNames.append("rgb")
        self.disparityOut.setStreamName("disp")
        self.queueNames.append("disp")
        self.leftOut.setStreamName("left")
        self.queueNames.append("left")
        self.NNout.setStreamName("detections")
        self.queueNames.append("detections")


        #Properties
        self.camRgb.setPreviewSize(self.inputSize)
        self.camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)                               # **NEW**
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)                 # **NEW**
        self.camRgb.setFps(self.fps)
        if self.downscaleColor: self.camRgb.setIspScale(2, 3)                               # **NOT IN MV3**
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = self.device.readCalibration2()
            lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if lensPosition:
                self.camRgb.initialControl.setManualFocus(lensPosition)
        except:
            raise
        self.left.setResolution(monoResolution)
        self.left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.left.setFps(self.fps)
        self.right.setResolution(monoResolution)
        self.right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        self.right.setFps(self.fps)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # LR-check is required for depth alignment
        self.stereo.setLeftRightCheck(True)
        self.stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        self.stereo.setOutputSize(self.left.getResolutionWidth(), self.left.getResolutionHeight())
        # self.stereo.setOutputSize(self.inputSize[0], self.inputSize[1])                                                 # **NEW**

        # Linking
        self.camRgb.preview.link(self.rgbOut.input)                                         # ** Changed to Preview from ISP**
        self.left.out.link(self.stereo.left)
        self.right.out.link(self.stereo.right)
        self.stereo.disparity.link(self.disparityOut.input)
        # self.camRgb.preview.link(spatialDetectionNetwork.input)
        # self.left.out.link(self.leftOut.input)
    
        # self.stereo.depth.link(spatialDetectionNetwork.inputDepth)
        # spatialDetectionNetwork.out.link(self.NNout.input)


    def runPipeline(self, processDetections, objectsCallback=None, displayResults=None, processImages=None):
        # Connect to device and start pipeline
        with self.device:
            self.device.startPipeline(self.pipeline)

            frameRgb = None
            frameDisp = None

            # Configure windows; trackbar adjusts blending ratio of rgb/depth
            rgbWindowName = "rgb"
            depthWindowName = "depth"
            blendedWindowName = "rgb-depth"
            cv2.namedWindow(rgbWindowName)
            cv2.namedWindow(depthWindowName)
            cv2.namedWindow(blendedWindowName)
            cv2.createTrackbar('RGB Weight %', blendedWindowName, int(rgbWeight*100), 100, updateBlendWeights)

            while True:
                latestPacket = {}
                for q in self.queueNames:
                    latestPacket[q] = None

                queueEvents = self.device.getQueueEvents(self.queueNames)
                for queueName in queueEvents:
                    packets = self.device.getOutputQueue(queueName).tryGetAll()
                    if len(packets) > 0:
                        latestPacket[queueName] = packets[-1]

                if latestPacket["rgb"] is not None:
                    frameRgb = latestPacket["rgb"].getCvFrame()
                    cv2.imshow(rgbWindowName, frameRgb)

                if latestPacket["left"] is not None:
                    frameLeft = latestPacket["left"].getCvFrame()
                    cv2.imshow("left", frameLeft)

                if latestPacket["disp"] is not None:
                    frameDisp = latestPacket["disp"].getFrame()
                    maxDisparity = self.stereo.initialConfig.getMaxDisparity()
                    # Optional, extend range 0..95 -> 0..255, for a better visualisation
                    if 1: frameDisp = (frameDisp * 255. / maxDisparity).astype(np.uint8)
                    # Optional, apply false colorization
                    if 1: frameDisp = cv2.applyColorMap(frameDisp, cv2.COLORMAP_HOT)
                    frameDisp = np.ascontiguousarray(frameDisp)
                    cv2.imshow(depthWindowName, frameDisp)

                # Blend when both received
                if frameRgb is not None and frameDisp is not None:
                    # Need to have both frames in BGR format before blending
                    if len(frameDisp.shape) < 3:
                        frameDisp = cv2.cvtColor(frameDisp, cv2.COLOR_GRAY2BGR)
                    blended = cv2.addWeighted(frameRgb, rgbWeight, frameDisp, depthWeight, 0)
                    cv2.imshow(blendedWindowName, blended)
                    frameRgb = None
                    frameDisp = None

                if cv2.waitKey(1) == ord('q'):
                    break


        # # Connect to device and start pipeline
        # with dai.Device(self.pipeline) as device:
        # # For now, RGB needs fixed focus to properly align with depth.
        # # This value was used during calibration
        #     try:
        #         calibData = device.readCalibration2()
        #         lensPosition = calibData.getLensPosition(dai.CameraBoardSocket.RGB)
        #         if lensPosition:
        #             self.camRgb.initialControl.setManualFocus(lensPosition)
        #     except:
        #         raise

        #     # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        #     previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        #     detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        #     depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        #     startTime = time.monotonic()
        #     counter = 0
        #     self.fps = 0

        #     if self.LaserDotProjectorCurrent is not None:
        #         device.setIrLaserDotProjectorBrightness(self.LaserDotProjectorCurrent)

        #     while True:
        #         self.inPreview = previewQueue.get()
        #         self.inDet = detectionNNQueue.get()
        #         self.depth = depthQueue.get()

        #         counter += 1
        #         current_time = time.monotonic()
        #         if (current_time - startTime) > 1:
        #             self.fps = counter / (current_time - startTime)
        #             counter = 0
        #             startTime = current_time

        #         self.frame = self.inPreview.getCvFrame()
        #         self.depthFrame = self.depth.getFrame()

        #         self.depthFrameColor = cv2.normalize(self.depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        #         self.depthFrameColor = cv2.equalizeHist(self.depthFrameColor)
        #         self.depthFrameColor = cv2.applyColorMap(self.depthFrameColor, cv2.COLORMAP_RAINBOW)

        #         detections = self.inDet.detections
        #         if len(detections) != 0:
        #             objects = processDetections(self, detections)
        #             if objects is None:
        #                 objects = []
        #         else:
        #             objects = []

        #         if processImages is not None:
        #             additionalObjects = processImages(self.frame, self.depthFrame, self.depthFrameColor)
        #             if additionalObjects is not None:
        #                 objects = objects + additionalObjects

        #         if objectsCallback is not None:
        #             objectsCallback(objects)

        #         self.displayDebug(device)

        #         if displayResults is not None:
        #             if displayResults(self.frame, self.depthFrameColor) == False:
        #                 return


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

                