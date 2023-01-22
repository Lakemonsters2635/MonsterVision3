#!/usr/bin/env python3

import json
import FRC
import OAK
import AprilTags
import cv2

PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200

aprilTags = AprilTags.AprilTags("tag16h5")

def processApriltags(imageFrame, depthFrame, depthFrameColor):
    return aprilTags.detect(imageFrame, depthFrame, depthFrameColor)

def processDetections(oak, detections):
    return oak.processDetections(detections)

def objectsCallback(objects):
    frc.writeObjectsToNetworkTable(json.dumps(objects))

def displayResults(frame, depthFrameColor):
    return frc.displayResults(frame, depthFrameColor)


frc = FRC.FRC()
frc.start(PREVIEW_WIDTH, PREVIEW_HEIGHT)

oak = OAK.OAK()
nnConfig = oak.read_nn_config()

spatialDetectionNetwork = oak.setupSDN(nnConfig)
oak.buildPipeline(spatialDetectionNetwork)


oak.runPipeline(processDetections, objectsCallback, displayResults, processApriltags)
i = 5