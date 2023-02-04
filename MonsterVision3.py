#!/usr/bin/env python3

import json
import FRC
import AprilTags
import cv2
import importlib
import Contours

# OAK = importlib.import_module("Gripper")            # Allows substitution of other pilelines!
OAK = importlib.import_module("MV3")            # Allows substitution of other pilelines!

PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200

aprilTags = AprilTags.AprilTags("tag16h5")  
contours = Contours.Contours()

def processExtra(imageFrame, depthFrame, depthFrameColor):
    objects = contours.detect(imageFrame, depthFrame, depthFrameColor)
    return objects.extend(aprilTags.detect(imageFrame, depthFrame, depthFrameColor))

def processDetections(oak, detections):
    return oak.processDetections(detections)

def objectsCallback(objects):
    frc.writeObjectsToNetworkTable(json.dumps(objects))

def displayResults(fullFrame, depthFrameColor, detectionFrame):
    return frc.displayResults(fullFrame, depthFrameColor, detectionFrame)


frc = FRC.FRC()
frc.start(PREVIEW_WIDTH, PREVIEW_HEIGHT)

oak = OAK.OAK(frc.LaserDotProjectorCurrent)
nnConfig = oak.read_nn_config()

spatialDetectionNetwork = oak.setupSDN(nnConfig)
oak.buildPipeline(spatialDetectionNetwork)


oak.runPipeline(processDetections, objectsCallback, displayResults, processExtra)
