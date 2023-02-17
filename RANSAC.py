import math
import random
import time
import numpy as np

TOLERANCE = 12.7            # Outliers are more than this distance (in mm) from the proposed plane
SUCCESS = .60               # If we get this percentage of inliers, declare victory
MAX_ITERATIONS = 600        # Give up after this many iterations and no success
MAX_POINTS_TO_FIT = 250         # Do the LS Fit on a randomly selected subset of points (for performance)


def findPlane(P10, P11, P12, P20, P21, P22, P30, P31, P32):

# Find the plane's normal vector by taking the cross product of two vectors between pairs of points

    (a0, a1, a2) = (P20-P10, P21-P11, P22-P12)
    (b0, b1, b2) = (P30-P10, P31-P11, P32-P12)
    (N0, N1, N2) = (a1*b2-a2*b1, a2*b0-a0*b2, a0*b1-a1*b0)

# Pick any point to compute D

    D = -(N0*P10 + N1*P11 + N2*P12)
    U = math.sqrt(N0*N0+N1*N1+N2*N2)

# Plane in Ax + By + Cz = D form, where (A, B, C) is the unit vector normal to the plane
    if U == 0:
        return (N0, N1, N2, D)
        
    return (N0/U, N1/U, N2/U, D/U)


def RANSAC(pointCloud, pointCount):

    if pointCount == 0:
        return None

    skipRANSAC = True

    start_time = time.process_time_ns()
   
# Begin RANSAC

    if not skipRANSAC:
        best = (0, 0, 0, 0)
        bestCount = 0

        for i in range(0, MAX_ITERATIONS):

    # Choose 3 random points and find the plane they define

            (P10, P11, P12) = pointCloud[random.randint(0, pointCount-1)]
            (P20, P21, P22) = pointCloud[random.randint(0, pointCount-1)]
            (P30, P31, P32) = pointCloud[random.randint(0, pointCount-1)]
            
            (A, B, C, D) = findPlane(P10, P11, P12, P20, P21, P22, P30, P31, P32)

    # Now loop over all points, deciding if each one fits the model or not.  We don't need to track the points
    # that fit.  We just need to count them

            inliers = 0
            found = False

            inrange = abs(A*pointCloud[:,0] + B*pointCloud[:,1] + C*pointCloud[:,2] +D) < TOLERANCE
            inliers = inrange.sum()

    # If we are better than previous best, record it.  If we have enough inliers, return

            if inliers > bestCount:
                bestCount = inliers
                best = (A, B, C, D)
                if (inliers / pointCount) >= SUCCESS:
                    found = True
                    break
                    # return (A, B, C, D)
        
    # Never found a good plane.  Give up.
    #     
        if not found:
            return None

    ransac_time = time.process_time_ns()

    # print()
    # print("A: {0:.2f}  B: {1:.2f}  C: {2:.2f}  D: {3:.2f}".format(A, B, C, D))

# Now do a least squares fit to the inliers.

# Compute distance from point to plane.  Dist = Ax + By + Cz + D
# This simplified formula works because (A, B, C) is a unit vector.

    if skipRANSAC:
        subArray = pointCloud[np.random.randint(pointCloud.shape[0], size=MAX_POINTS_TO_FIT)]
    else:
        pointsToFit = abs(A*pointCloud[:,0] + B*pointCloud[:,1] + C*pointCloud[:,2] +D) < TOLERANCE
        num = pointsToFit.sum()
        subArray = pointCloud[pointsToFit]
        if (MAX_POINTS_TO_FIT < num):
            subArray = subArray[np.random.randint(subArray.shape[0], size=MAX_POINTS_TO_FIT)]

    tmp_b = np.matrix(subArray[:,2]).T
    tmp_a = np.matrix(np.insert(subArray[:,0:2], 2, 1, axis=1))

    fit = (tmp_a.T * tmp_a).I * tmp_a.T * tmp_b

    lsfit_time = time.process_time_ns()

    A = fit[0,0]
    B = fit[1,0]
    C = 1
    D = -fit[2,0]

    # print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))

    U = math.sqrt(A*A+B*B+C*C)
    if U == 0:
        return (A, B, C, D)

    A /= U
    B /= U
    C /= U
    D /= U

    # xAngle = math.pi - math.atan2(C, B)
    # yAngle = math.pi - math.atan2(C, A)
    # print("A: {0:.2f}  B: {1:.2f}  C:{2:.2f}  D: {3:.2f}  X: {4:.2f}  Y:  {5:.2f}".format(A, B, C, D, xAngle*180/math.pi,  yAngle*180/math.pi)) 
    # print("X: {0:.2f}".format    print(f"ransac: {ransac_time/1000000.0:8.2f} ms")

    total_time = lsfit_time - start_time
    lsfit_time -= ransac_time
    ransac_time -= start_time

    # print(f"ransac:    {ransac_time/1000000.0:8.2f} ms")
    # print(f"lsfit({num}): {lsfit_time/1000000.0:8.2f} ms")
    # print(f"total:    {total_time/1000000.0:8.2f}ms")

    return (A, B, C, D)
