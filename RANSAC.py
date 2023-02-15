import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt

TOLERANCE = 12.7            # Outliers are more than this distance (in mm) from the proposed plane
SUCCESS = .60               # If we get this percentage of inliers, declare victory
MAX_ITERATIONS = 300        # Give up after this many iterations and no success

PLOTIT = False

iter = 0

def poseAngleFromVector(x, y):

# get the normalized components of the vector

    U = math.sqrt(x*x + y*y)
    if U == 0:
        return None

    x /= U
    y /= U

# The angle is now the arccos(y)

    angle = math.acos(y)
    if angle > math.pi/2:
        angle -= math.pi/2

    return angle


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

    start_time = time.monotonic_ns()
    
# Begin RANSAC

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

        for (p0, p1, p2) in pointCloud:

# Compute distance from point to plane.  Dist = Ax + By + Cz + D
# This simplified formula works because (A, B, C) is a unit vector.

            Dist = abs(A*p0 + B*p1 + C*p2 + D)
            
            if Dist < TOLERANCE:
                inliers += 1

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

    ransac_time = time.monotonic_ns()

    # print()
    # print("A: {0:.2f}  B: {1:.2f}  C: {2:.2f}  D: {3:.2f}".format(A, B, C, D))

# Now do a least squares fit to the inliers.

    tmp_A = []
    tmp_b = []

    if PLOTIT:
        xs = []
        ys = []
        zs = []

    for (p0, p1, p2) in pointCloud:

# Compute distance from point to plane.  Dist = Ax + By + Cz + D
# This simplified formula works because (A, B, C) is a unit vector.

        Dist = abs(A*p0 + B*p1 + C*p2 + D)
        
        if Dist < TOLERANCE:
            tmp_A.append([p0, p1, 1])
            tmp_b.append(p2)
            if PLOTIT:
                xs.append(p0)
                ys.append(p1)
                zs.append(p2)

    b = np.matrix(tmp_b).T
    AA = np.matrix(tmp_A)

# Manual solution
    fit = (AA.T * AA).I * AA.T * b

    lsfit_time = time.monotonic_ns()

    if PLOTIT:
        # plot raw data
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(xs, ys, zs, color='b')
# plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                        np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
        ax.plot_wireframe(X,Y,Z, color='k')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()


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

    # xAngle = poseAngleFromVector(B, C) 
    # yAngle = poseAngleFromVector(A, C)
    # print("A: {0:.2f}  B: {1:.2f}  C:{2:.2f}  D: {3:.2f}  X: {4:.2f}  Y:  {5:.2f}".format(A, B, C, D, xAngle*180/math.pi,  yAngle*180/math.pi)) 
    # print("X: {0:.2f}".format(xAngle*180/math.pi))

    total_time = lsfit_time - start_time
    lsfit_time -= ransac_time
    ransac_time -= start_time

    # print(f"ransac: {ransac_time/1000000.0:8.2f} ms")
    # print(f"lsfit: {lsfit_time/1000000.0:8.2f} ms")
    # print(f"total: {total_time/1000000.0:8.2f}ms")

    global iter
    iter += 1

    # if iter > 100:
    #     fo = open("pointCloud.txt", "w")
    #     fo.write("%d"%pointCount)
    #     for p in pointCloud:   
    #         fo.write("%f %f %f"%p)
    #     fo.close()

    return (A, B, C, D)
