"""
CMod Project B: Simulation of
N particles interacting through a LJ pair potential.

Produce a plot of the particles to be visualised using VMD and investigate
the equilibrium properties of the simulated system.
"""

import sys
import numpy as np
import copy
import matplotlib.pyplot as pyplot
import MDUtilities as MD
import LJFunctions as LJ
from Particle3D import Particle3D

# Read names of output file and input file from command line
if len(sys.argv) != 3:
    print "Wrong number of arguments."
    print "Usage: " + sys.argv[0] + " <output file> + <input file>"
    quit()
else:
    outfileName = sys.argv[1]
    infileName = sys.argv[2]

# Open output file for writing
outfile = open(outfileName, "w")
# Open input file for reading
infile = open(infileName, "r")
# Open output file for particle energy data for writing
energyoutfile = open("energyoutfile.dat", "w")
# Open output file for MSD data
MSDoutfile = open("MSDoutfile.dat", "w")
# Open output file for RDF data
RDFoutfile = open("RDFoutfile.dat", "w")

# Read particle data from input file
line = infile.readline()
tokens = line.split()

# Number of particles
numPart = int(tokens[0])

# Number of Simulation steps
numSteps = int(tokens[1])

# Frequency of file output during Sim
freqOut = int(tokens[2])

# Timestep
dt = float(tokens[3])

# Reduced temperature
temp = float(tokens[4])

# Reduced density
rho = float(tokens[5])

# LJ cut off distance - for pairwise force and energy calculations
LJcut = float(tokens[6])

# Close input file
infile.close()

# Set up N particles with placeholder values
particleList = []
for i in range(0, numPart):
    particleLabel = "particle" + str(i + 1)
    particleList.append(Particle3D(np.array([0, 0, 0]), np.array([0, 0, 0]), 1.0, particleLabel))

# Set up initial positions and get the simulation box dimensions
boxDim = MD.setInitialPositions(rho, particleList)

# Set up list to store the particles at their initial positions - for the MSD calculation
initialPosPart = []
for p in particleList:
    # Create a copy of the initial particle to put in the list
    p1 = copy.copy(p)
    # Add the particle to the end of the list
    initialPosPart.append(p1)

# Set up initial velocities
MD.setInitialVelocities(temp, particleList)

# Set initial force array
force = Particle3D.particleListForce(particleList, LJcut, boxDim)

# Set initial time
time = 0.

# Set timestep for VMD file
timestep = 1

# Set RDF data collection variables
dr = 0.0005 * boxDim[0]
rmax = 0.5 * boxDim[0]
numDen = float(len(particleList)) / (boxDim[0] ** 3)

# Set empty lists for time data, time averaged Histogram values, MSD data and all energy data
histList = []
rValues = []
tValues = []
MSDlist = []
totalElist = []
KElist = []
PElist = []

# Create array of r values for RDF data
for i in range(int(rmax / dr)+1):
    rValues.append(float(i * dr))

# A counter to control the frequency of file output in the simulation
counter = 1

# Start the time integration loop

for i in range(numSteps):
    print(i)
    # Update all particle positions
    Particle3D.positionUpdateList(particleList, force, dt)
    # Update force
    force_new = Particle3D.particleListForce(particleList, LJcut, boxDim)
    # Update particle velocities, based on average
    # of current and new forces
    Particle3D.velocityUpdateList(particleList, 0.5 * (force + force_new), dt)
    # Places the particles back within the box boundaries if they move out
    LJ.checkPBC(boxDim, particleList)

    # Reset force variable
    force = force_new

    # Increase time
    time += dt

    # Sends data to output files if the simulation is on its nth timestep
    # n is the desired frquency of output to output files
    if counter == freqOut:

        # Update total, kinetic and potential energy for all particles
        totalKE = Particle3D.totalKineticEnergy(particleList)
        totalPE = Particle3D.totalPotentialEnergy(particleList, LJcut, boxDim)
        totalE = totalKE + totalPE

        # Writes the energy data to an output file and update all lists
        energyoutfile.write("{0:f} {1:f} {2:f} {3:f}\n".format(totalE, totalKE, totalPE, time))
        totalElist.append(totalE)
        KElist.append(totalKE)
        PElist.append(totalPE)

        # Update MSD
        MSD = LJ.MeanSD(particleList, initialPosPart, boxDim)

        # Write MSD data to an output file and update list
        MSDoutfile.write("{0:f} {1:f}\n".format(MSD, time))
        MSDlist.append(MSD)

        # Calculate RDF for this timestep and add it to the histogram list
        pDist = LJ.rij(particleList, boxDim)
        hist = LJ.bin(pDist, dr, rmax)
        normHist = LJ.normalization(hist, dr, numDen)
        histList.append(normHist)

        # Add new VMD entry
        Particle3D.entryVMD(particleList, timestep, outfile)

        # Append current time to list of time values
        tValues.append(time)

        timestep += 1
        counter = 1
    else:
        counter += 1

# Convert the list of histograms into a numpy array
histList = np.array(histList)
# Time average and normalize the histograms
timeAveRDF = np.sum(histList, axis=0) / (numPart * numDen * len(histList))
# Write the data to the RDF output file
for i in range(len(timeAveRDF)):
    RDFoutfile.write("{0:f} {1:f}\n".format(timeAveRDF[i], rValues[i]))

# Plot all observables and display after the simulation has run

# Produce RDF plot against r
RDFplot = pyplot.plot(rValues, timeAveRDF, "r")
pyplot.title("RDF against distance r")
pyplot.xlabel("r (reduced length)")
pyplot.ylabel("RDF(r)")
pyplot.show(RDFplot)

# Produce MSD plot against time
MSDPlot = pyplot.plot(tValues, MSDlist, "r")
pyplot.title("Mean Squared Displacement against time")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Mean Squared Displacement (reduced length)")
pyplot.show(MSDPlot)

# Produce Energy (total, kinetic, potential) plot agains time
EnergyPlot = pyplot.plot(tValues, totalElist, "r", tValues, PElist, "b", tValues, KElist, "g")
pyplot.title("Total, Kinetic and Potential energy against time")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Total Energy (red), Kinetic Energy (green), Potential Energy (blue) (J)")
pyplot.show(EnergyPlot)

# Close output files
RDFoutfile.close()
MSDoutfile.close()
energyoutfile.close()
outfile.close()
