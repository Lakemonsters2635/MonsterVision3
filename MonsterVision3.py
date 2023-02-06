#!/usr/bin/env python3

from ast import List
import json
import threading
import time
import FRC
import AprilTags
import cv2
import importlib
import Contours
import DAI
import depthai as dai

def processExtra1(imageFrame, depthFrame, depthFrameColor, contours):
    return contours.detect(imageFrame, depthFrame, depthFrameColor)

def processExtraD(imageFrame, depthFrame, depthFrameColor, aprilTags):
    return aprilTags.detect(imageFrame, depthFrame, depthFrameColor)

def processDetections(oak, detections):
    return oak.processDetections(detections)

def objectsCallback(objects, cam):
    global frc
    frc.writeObjectsToNetworkTable(json.dumps(objects), cam)

def displayResults(fullFrame, depthFrameColor, detectionFrame, cam):
    return frc.displayResults(fullFrame, depthFrameColor, detectionFrame, cam)




def runOAK1(mxid, cam):
    OAK = importlib.import_module("Gripper")            # Allows substitution of other pilelines!
    contours = Contours.Contours()
    oak = OAK.OAK(None)
    nnConfig = oak.read_nn_config()

    spatialDetectionNetwork = oak.setupSDN(nnConfig)
    oak.buildPipeline(spatialDetectionNetwork)

    oak.runPipeline(processDetections, objectsCallback, displayResults, processExtra1, cam, contours)
    return

def runOAKD(mxid, cam):
    OAK = importlib.import_module("MV3")            # Allows substitution of other pilelines!
    aprilTags = AprilTags.AprilTags("tag16h5")  
    oak = OAK.OAK(frc.LaserDotProjectorCurrent)
    nnConfig = oak.read_nn_config()

    spatialDetectionNetwork = oak.setupSDN(nnConfig)
    oak.buildPipeline(spatialDetectionNetwork)

    oak.runPipeline(processDetections, objectsCallback, displayResults, processExtraD, cam, aprilTags)
    return

PREVIEW_WIDTH = 200
PREVIEW_HEIGHT = 200

frc = FRC.FRC()
frc.start(PREVIEW_WIDTH, PREVIEW_HEIGHT)


OAK_D_MXID = None
OAK_1_MXID = None

# devices = DAI.DAI.getDevices()

# for c in devices:
#     if c["cameras"] == 1:
#         if OAK_1_MXID is None:
#             OAK_1_MXID = c["mxid"]
#         else:
#             print(f"Found multiple OAK-1 devices.  Using {OAK_1_MXID}")
#     elif c["cameras"] == 3:
#         if OAK_D_MXID is None:
#             OAK_D_MXID = c["mxid"]
#         else:
#             print(f"Found multiple OAK-D devices.  Using {OAK_D_MXID}")
#     else:
#         print(f'Found device {c["mxid"]} having {c["cameras"]}.  This is unusual.')

infos = dai.DeviceBootloader.getAllAvailableDevices()
OAK_D_MXID = "14442C105129C6D200"
OAK_1_MXID = "14442C10E1474FD000"

def checkCam(infos, mxid):
    for i in infos:
        if mxid == i.mxid: return True
    return False

if not checkCam(infos, OAK_D_MXID): OAK_D_MXID = None
if not checkCam(infos, OAK_1_MXID): OAK_1_MXID = None


print("Using cameras:")
if OAK_D_MXID is not None: print(f"{OAK_D_MXID} OAK-D")
if OAK_1_MXID is not None: print(f"{OAK_1_MXID} OAK-1")

thread1 = None
threadD = None

if OAK_1_MXID is not None:
    thread1 = threading.Thread(target=runOAK1, args=(OAK_1_MXID, "Gripper", ))
    thread1.start()

if OAK_D_MXID is not None:
    threadD = threading.Thread(target=runOAKD, args=(OAK_D_MXID, "Chassis", ))
    threadD.start()

# Wait for both threads to complete (actually, should never happen!)
# time.sleep(36000)
if thread1 is not None: thread1.join()
if threadD is not None: threadD.join(5)


