import apriltag
import cv2
import math
import numpy as np

INCHES_PER_MILLIMETER = 39.37 / 1000

class AprilTags:
    # Required information for calculating spatial coordinates on the host
    monoFOV = np.deg2rad(72)
    tanHalfHFOV = math.tan(monoFOV / 2.0)

    penucheFactorM = 0.785
    penucheFactorB = 10 * 25.4

    def calc_tan_angle(self, offset, depthWidth):
        return offset * self.tanHalfHFOV / depthWidth


    def mapXCoord(self, Small):
        return int((Small + self.xMin) * self.Scale)


    def mapYCoord(self, Small):
        return int((Small + self.yMin) * self.Scale)

    def mapCoords(self,pt):
        return (self.mapXCoord(pt[0]), self.mapYCoord(pt[1]))


    # This code assumes depth symmetry around the centroid

    # Calculate spatial coordinates from depth map and bounding box (ROI)

    def calc_spatials(self, bbox, centroidX, centroidY, depth, inputShape, averaging_method=np.median):
        xmin, ymin, xmax, ymax = bbox
        if xmin > xmax:  # bbox flipped
            xmin, xmax = xmax, xmin
        if ymin > ymax:  # bbox flipped
            ymin, ymax = ymax, ymin

        a = self.mapCoords((256, 256))            

        xmin = self.mapXCoord(xmin)
        xmax = self.mapXCoord(xmax)
        ymin = self.mapYCoord(ymin)
        ymax = self.mapYCoord(ymax)

        if xmin == xmax or ymin == ymax: # Box of size zero
            return None

        # Calculate the average depth in the ROI.
        depthROI = depth[ymin:ymax, xmin:xmax]
        averageDepth = averaging_method(depthROI)

        cX = self.mapXCoord(centroidX)
        cY = self.mapYCoord(centroidY)
        bb_x_pos = cX - int(depth.shape[1] / 2)
        bb_y_pos = cY - int(depth.shape[0] / 2)

        # angle_x = calc_angle(bb_x_pos, depthWidth)
        # angle_y = calc_angle(bb_y_pos, depthWidth)
        tanAngle_x = self.calc_tan_angle(bb_x_pos, inputShape)
        tanAngle_y = self.calc_tan_angle(bb_y_pos, inputShape)

        # z = averageDepth
        # print(f"z: {averageDepth* 39.37 / 1000}")
        z = averageDepth * self.penucheFactorM + self.penucheFactorB
        # print(f"modified z: {(averageDepth * self.penucheFactorM + self.penucheFactorB)* 39.37 / 1000}")
        x = z * tanAngle_x
        y = -z * tanAngle_y

        return (x,y,z)


        
    def __init__(self, tagFamilies, rgbHFOV = None):
        options = apriltag.DetectorOptions(families="tag16h5")
        self.detector = apriltag.Detector(options)
        if rgbHFOV is not None:
            self.rgbHFOV = np.deg2rad(rgbHFOV)
            self.tanHalfHFOV = math.tan(self.rgbHFOV / 2.0)
       

    def drawBoundingBox(self, frame, ptA, ptB, ptC, ptD, color, lineWidth):
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)



    def detect(self, imageFrame, depthFrame, depthFrameColor):
        xScale = depthFrame.shape[1] / imageFrame.shape[1]
        yScale = depthFrame.shape[0] / imageFrame.shape[0]

        if xScale > yScale:
            dim = (int(imageFrame.shape[0]*depthFrame.shape[1]/depthFrame.shape[0]), int(imageFrame.shape[0]))
            self.xMin = int((dim[0] - imageFrame.shape[1])/2)
            # self.xMax = int(dim[0] - xmin)
            self.yMin = 0
            self.yMax = imageFrame.shape[0]
            self.Scale = yScale
        else:
            dim = (int(imageFrame.shape[1]), int(imageFrame.shape[1]*depthFrame.shape[0]/depthFrame.shape[1]))
            self.yMin = int((dim[1] - imageFrame.shape[0])/2)
            # self.yMax = int(dim[1] - ymin)
            self.xMin = 0
            self.xMax = imageFrame.shape[1]
            self.Scale = xScale

        gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray)

        objects = []

        # if len(results) > 4:
        #     i = 0

        # loop over the AprilTag detection results
        for r in results:
            if r.hamming != 0:
                continue
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            res = self.calc_spatials((ptA[0], ptA[1], ptC[0], ptC[1]), cX, cY, depthFrame, depthFrame.shape[0])
            if res == None:
                continue
            (atX, atY, atZ) = res
            atX = round((atX * INCHES_PER_MILLIMETER), 1)
            atY = round((atY * INCHES_PER_MILLIMETER), 1)
            atZ = round((atZ * INCHES_PER_MILLIMETER), 1)

            # draw the bounding box of the AprilTag detection
            self.drawBoundingBox(imageFrame, ptA, ptB, ptC, ptD, (0, 255, 0), 2)
            self.drawBoundingBox(depthFrameColor, self.mapCoords(ptA), self.mapCoords(ptB), self.mapCoords(ptC), self.mapCoords(ptD), (0, 0, 0), 4)
            # self.drawBoundingBox(depthFrameColor, ptA, ptB, ptC, ptD, (255, 0, 255), 2)

            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(imageFrame, (cX, cY), 5, (0, 0, 255), -1)
            wd = abs(ptC[0] - ptA[0])
            ht = abs(ptC[1] - ptA[1])
            lblX = int(cX - wd/2)
            lblY = int(cY - ht/2)
            # draw the tag family on the image
            tagID= '{}: {}'.format(r.tag_family.decode("utf-8"), r.tag_id)
            color = (255, 0, 0)
            cv2.putText(imageFrame, tagID, (lblX, lblY - 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"X: {atX} in", (lblX, lblY - 45), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"Y: {atY} in", (lblX, lblY - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(imageFrame, f"Z: {atZ} in", (lblX, lblY - 15), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            objects.append({"objectLabel": tagID, "x": atX, "y": atY, "z": atZ})

        return objects


