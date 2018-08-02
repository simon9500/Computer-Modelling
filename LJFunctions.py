"""
CMod Project B: extra methods for applying Periodic Boundary Conditions and finding equilibrium properties of a LJ system
"""

import math
import numpy as np
from Particle3D import Particle3D


# Method to move particle back into the simulation box if they move outside - using Periodic Boundary Conditions
def checkPBC(boxDim, particleList):
    # Loop through all particles
    for p in particleList:
        # Loop through all 3 axes
        for i in range(3):
            # Check if particle has gone out the boundary on the the ith (x,y,z) axis and move it to its correct
            # position
            p.position[i] = p.position[i] % boxDim[i]


# Calculates the Mean Square Displacement of the system
def MeanSD(particleList, initialPosPart, boxDim):
    y = 0.
    for i in range(len(particleList)):
        x = Particle3D.scalarSeparation(particleList[i], initialPosPart[i], boxDim) ** 2
        y += x
    MSD = (1. / len(particleList)) * y
    return MSD


# Finds the distances between all particle pairs and returns them in a list
def rij(particleList, boxDim):
    pDistances = []
    for i, el in enumerate(particleList):
        for j, al in enumerate(particleList):
            if i < j:
                pDistances.append(Particle3D.scalarSeparation(el, al, boxDim))
    return pDistances


# Returns the (un-normalized) histogram as a list
def bin(pDist, dr, rmax):
    imax = rmax / dr
    histogram = np.zeros(int(imax)+1)
    for j in pDist:
        if j < rmax:
            histogram[int(j / dr)] += 1.
    return histogram


# Normalizes the histogram and returns it as a list
def normalization(hist, dr, numDen):
    normHist = np.zeros(len(hist))
    for i, el in enumerate(hist):
        normHist[i] = el / (4. * math.pi * numDen * dr * ((i - 1. / 2.) * dr) ** 2)
    return normHist
