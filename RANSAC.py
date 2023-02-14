import math
import random

TOLERANCE = 12.7            # Outliers are more than this distance (in mm) from the proposed plane
SUCCESS = .60               # If we get this percentage of inliers, declare victory
MAX_ITERATIONS = 300        # Give up after this many iterations and no success

def findPlane(p1, p2, p3):

# Find the plane's normal vector by taking the cross product of two vectors between pairs of points
    (P10, P11, P12) = p1
    (P20, P21, P22) = p2
    (P30, P31, P32) = p3

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

# Begin RANSAC

    best = (0, 0, 0, 0)
    bestCount = 0

    for i in range(0, MAX_ITERATIONS):

# Choose 3 random points and find the plane they define

        (P10, P11, P12) = pointCloud[random.randint(0, pointCount-1)]
        (P20, P21, P22) = pointCloud[random.randint(0, pointCount-1)]
        (P30, P31, P32) = pointCloud[random.randint(0, pointCount-1)]
        
        (A, B, C, D) = findPlane((P10, P11, P12), (P20, P21, P22), (P30, P31, P32))

# Now loop over all points, deciding if each one fits the model or not.  We don't need to track the points
# that fit.  We just need to count them

        inliers = 0

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
                return (A, B, C, D)
    
# Never found a good plane.  Give up.
#     
    return None
