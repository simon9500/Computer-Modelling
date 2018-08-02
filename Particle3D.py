"""
 CMod Ex3: Particle3D, a class to describe 3D particles
"""

import sys
import math
import numpy as np
import copy


class Particle3D(object):
    # Initialise a Particle3D instance
    def __init__(self, pos, vel, mass, label):
        self.position = np.array(pos)
        self.velocity = np.array(vel)
        self.mass = mass
        self.label = label

    # Formatted output as String
    def __str__(self):
        return str(self.label) + " " + str(self.position[0]) + " " + str(self.position[1]) + " " + str(self.position[2])

    # Kinetic energy
    def kineticEnergy(self):
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    # Returns the magnitude of the velocity of the particle
    def magVel(self):
        return np.dot(self.velocity, self.velocity) ** (1.0 / 2.0)

    # Returns the magnitude of the position of the particle
    def magPos(self):
        return np.dot(self.position, self.position) ** (1.0 / 2.0)

    # Time integration methods
    # First-order velocity update
    def leapVelocity(self, dt, force):
        self.velocity = self.velocity + dt * force / self.mass

    # First-order position update
    def leapPos1st(self, dt):
        self.position += dt * self.velocity

    # Second-order position update
    # Contains functionality to implement Periodic Boundary Conditions
    def leapPos2nd(self, dt, force):
        self.position = self.position + dt * self.velocity + 0.5 * dt ** 2 * force / self.mass

    # Create a particle from a file
    @staticmethod
    def createParticle(inFile):
        line = inFile.readline()
        tokens = line.split()
        velocity = np.array(float(tokens[0]), float(tokens[1]), float(tokens[2]))
        line = inFile.readline()
        tokens = line.split()
        position = np.array(float(tokens[0]), float(tokens[1]), float(tokens[2]))
        line = inFile.readline()
        mass = float(line)
        line = inFile.readline()
        label = line
        return Particle3D(position, velocity, mass, label)

    # Compute the vector separation of 2 particles
    # Includes functionality to find the closest image of a particle using the minimum image convention
    @staticmethod
    def vectorSeparation(p1, p2, boxDim):
        separation = np.zeros(3)
        for i in range(3):
            separation[i] = p1.position[i] - p2.position[i]
            separation[i] -= int(separation[i] / (boxDim[i] / 2.0)) * boxDim[i]
        return separation

    # Compute the magnitude of the vector separation between 2 particles
    @staticmethod
    def scalarSeparation(p1, p2, boxDim):
        return np.linalg.norm(Particle3D.vectorSeparation(p1, p2, boxDim))

    # Compute the LJ force between 2 particles
    @staticmethod
    def particleForce(p1, p2, rc, boxDim):
        r = Particle3D.scalarSeparation(p1, p2, boxDim)
        if r > rc:
            return 0.
        else:
            if r == 0:
                return 0.
            else:
                return (48 * (1. / (r ** 14) - 1. / ((r ** 8) * 2.))) * Particle3D.vectorSeparation(p1, p2, boxDim)

    # Compute the LJ potential energy between 2 particles
    @staticmethod
    def particlePotEnergy(p1, p2, rc, boxDim):
        r = Particle3D.scalarSeparation(p1, p2, boxDim)
        if r > rc:
            return 0.
        else:
            if r == 0:
                return 0.
            else:
                return 4. * ((1. / r ** 12) - (1. / r ** 6))

    # Updates the velocity of all particles in a list
    # using the LJ force and leapVelocity method
    @staticmethod
    def velocityUpdateList(pList, force, dt):
        for i, el in enumerate(pList):
            el.leapVelocity(dt, force[i])

    # Updates the position of all particles in a list
    # using the 2nd order position update and LJ force
    @staticmethod
    def positionUpdateList(pList, force, dt):
        for i, el in enumerate(pList):
            el.leapPos2nd(dt, force[i])

    # Calculate the total LJ force acting on each particle
    @staticmethod
    def particleListForce(pList, rc, boxDim):
        # Initialise force array
        force = np.empty([len(pList), 3])
        # Loop through all particles
        for j, al in enumerate(pList):
            # Initialise force on current particle
            x = np.zeros(3)
            # Loop through all other particles to calculate force pairs
            for i, el in enumerate(pList):
                if i != j:
                    x += Particle3D.particleForce(al, el, rc, boxDim)
            force[j] = x
        return force

    # Calculates the total kinetic energy for all particles in a list
    @staticmethod
    def totalKineticEnergy(pList):
        kinetic = 0.
        # Loop through all particles
        for el in pList:
            kinetic += el.kineticEnergy()
        return kinetic

    # Calculates the total potential energy by considering the LJ potential between every pair
    @staticmethod
    def totalPotentialEnergy(pList, rc, boxDim):
        totalPotential = 0.
        for i, el in enumerate(pList):
            for j, al in enumerate(pList):
                # Conditional statement to ensure interactions are not double counted
                if i < j:
                    totalPotential += Particle3D.particlePotEnergy(el, al, rc, boxDim)
        return totalPotential

    # Adds data for one timestep to an output file to be visualised in VMD
    @staticmethod
    def entryVMD(pList, timestep, outFile):
        # Write number of particles
        outFile.write(str(len(pList)) + "\n")
        # Write timestep
        outFile.write("Point = " + str(timestep) + "\n")
        # Loop through all particles
        for i, p in enumerate(pList):
            # Write label and position data (in float format) for a single particle
            outFile.write(str(p.label) + " {0:f} {1:f} {2:f}\n".format(p.position[0], p.position[1], p.position[2]))
