import random
import numpy as np
from itertools import product 
import time
import math

from fish import *

class swarm:
    nrVectorStates=5
    maxAngle = 4.*np.pi/180.
    def __init__(self, N, numNN, numdimensions, movementType, initType, _psi=-1,
    _nu = 1.,seed=10, _rRepulsion = 0.6, _delrOrientation=2.0, _delrAttraction=15.0, 
    _alpha=4.5, _initcircle = +7.0, _f=0.1, _height= +3., _emptzcofactor=+0.5):
        random.seed(seed)
        self.seed=seed
        #number of dimensions of the swarm
        self.dim = numdimensions
        # number of fish
        self.N = N
        # number of nearest neighbours
        self.numNearestNeighbours = numNN
        # type of movement the fish followk
        self.movType  = movementType
        self.rRepulsion = _rRepulsion
        self.rOrientation = _rRepulsion+_delrOrientation
        self.rAttraction = _rRepulsion+_delrOrientation+_delrAttraction
        self.alpha = _alpha
        self.initializationType = initType
        # Maximal number of initializations
        self.tooManyInits=False
        self.maxInits=5000
        self.speed=3.
        self.angularMoments = []
        self.polarizations = []
       
        # this keeps tracks of the angle between velocities of different fishes
        #self.anglesVelMat = np.empty(shape=(self.N,self.N),    dtype=float)
        # In case we have a cylinder we want to control its height
        self.height = _height
        #extra parameter to control polarization see Gautrais et al. "Initial polarization"
        self.psi = _psi
        # parameter to weigh how important the attraction is over the orientation
        self.nu = _nu
        # Circle parameter as described in Gautrais et al. page 420 top right
        self.f = _f
        self.initialCircle = pow(self.N, 1/self.dim)*self.f*self.rAttraction
        # self.initialCircle = _initcircle
        # boolean to see if we want to plot the shortest distance of the fishes
        self.plotShortestDistance = False
        # In case of a ring disposition what % of the initial
        # circle shuld be empty
        self.emptycorecofactor = _emptzcofactor
        self.emptyray = self.initialCircle * self.emptycorecofactor
        self.extended = True
        lonefish = True
        trycounter = 0
        while(lonefish):
            # placement on a grid
            if(self.initializationType == 0):
                self.fishes = self.randomPlacementNoOverlap(seed)
            elif (self.initializationType in np.array([1, 2])):
                self.fishes = self.initInSphereorCyl()
            else:
                print("[swarm] Unknown initialization type, please choose a number between 0 and 2")
                exit(0)
            
            lonefish = self.noperceivefishinit(self.fishes)
            trycounter += 1 
            # print("number of initializations: ", trycounter)
            if(trycounter == self.maxInits):
                print("over ", trycounter, " initializations")
                self.printstate()
                self.tooManyInits=True
                lonefish = False
        self.angularMoments.append(self.computeAngularMom())
        self.polarizations.append(self.computePolarisation())
        

    def initFromLocationsAndDirections(self, locations, directions):
        assert locations.shape == directions.shape
        assert locations.shape[0] == self.N
        assert locations.shape[1] == self.dim

        for idx, loc in enumerate(locations):
            self.fishes[idx] = fish(loc, directions[idx], self.dim, self.psi)

        self.angularMoments.append(self.computeAngularMom())
        self.polarizations.append(self.computePolarisation())
        self.preComputeStates()
 


    """ random placement on a grid """
    # NOTE the other two papers never start on grids but they start on sphere like structures
    def randomPlacementNoOverlap(self, seed):
        # number of gridpoints per dimension for initial placement
        M = int( pow( self.N, 1/self.dim ) )
        V = M+1
        # grid spacing ~ min distance between fish
        # NOTE this 0.7 comes from a few test Daniel run
        dl = 0.7
        # maximal extent
        L = V*dl
        # generate random permutation of [1,..,V]x[1,..,V]x[1,..,V]
        perm = list(product(np.arange(0,V),repeat=self.dim))
        assert self.N < len(perm), "More vertices required to generate random placement"
        random.Random( seed ).shuffle(perm)
    
        # place fish
        fishes = np.empty(shape=(self.N,), dtype=fish)
        location = np.zeros(shape=(self.N,self.dim), dtype=float)

        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)
        
        if(self.dim == 2):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim, self.psi, maxAngle=self.maxAngle, speed=self.speed)
 
        elif(self.dim == 3):
            for i in range(self.N):
                location = np.array([perm[i][0]*dl, perm[i][1]*dl, perm[i][2]*dl]) - L/2
                initdirect=reffish.randUnitDirection()
                fishes[i] = fish(location, initdirect, self.dim, self.psi,maxAngle=self.maxAngle, speed=self.speed)
       
        # return array of fish
        return fishes

    # different explanations of how to generate random unit distr https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
    # on the unit speher$
    """Generate N random uniform points within a sphere"""
    def initInSphereorCyl(self):
        # reference fish which is useless basically
        reffish = fish(np.zeros(self.dim),np.zeros(self.dim), self.dim, self.psi)
        fishes = np.empty(shape=(self.N, ), dtype=fish)
        # based on https://stackoverflow.com/questions/9048095/create-random-number-within-an-annulus
        # Normalizing costant
        r_max = self.initialCircle
        normFac = 1./(r_max*r_max)

        if self.initializationType==1:
            for i in range(self.N):
                
                initdirect= reffish.randUnitDirection() #vec/np.linalg.norm(vec)
                if(self.dim == 2):
                    r = np.sqrt(random.uniform(0,1)/normFac)
                    theta = np.random.uniform() * 2 * np.pi
                    vec = np.array([r * np.cos(theta), r * np.sin(theta)])
 
                if(self.dim == 3):
                    phi = random.uniform(0,2*np.pi)
                    costheta = random.uniform(-1,1)
                    u = random.uniform(0,1)
                    theta = np.arccos( costheta )
                    r = r_max * np.cbrt(u)
                    
                    x = r * np.sin( theta ) * np.cos( phi )
                    y = r * np.sin( theta ) * np.sin( phi )
                    z = r * np.cos( theta )
                    vec = np.array([x, y, z])

                fishes[i] = fish(vec, initdirect, self.dim, self.psi, speed=self.speed, maxAngle=self.maxAngle, randomMov=(self.movType == 1))
        
        if self.initializationType==2:
            for k in range(self.N):
                r = np.sqrt(2*random.uniform(0,1)/normFac + r_min*r_min)
                theta = np.random.uniform() * 2 * np.pi
                z = np.random.uniform(-height/2, +height/2)
                location = np.array([r * np.cos(theta), r * np.sin(theta), z])
                projxy = np.array([location[0],location[1],0])
                initdirect = projxy/np.linalg.norm(projxy)
                initdirect = reffish.applyrotation(initdirect, np.pi/2, twodproj=True)
                fishes[k] = fish(location, initdirect, self.dim, self.psi, maxAngle=self.maxAngle, speed=self.speed)

        return fishes

    """Boolean function that checks that in the fishes list all fishes perceive at least one other fish"""
    def noperceivefishinit(self, fishes):
        for i, fish in enumerate(fishes):
            _, distances, angles, _, _, _, _, _ = self.retpreComputeStates(fishes)
            repellTargets, orientTargets, attractTargets = self.retturnrep_or_att(i, fish, angles, distances)

            # Check if the the repellTargets, orientTargets, attractTargets are empty
            if(not any(repellTargets) and not any(orientTargets) and not any(attractTargets)):
                return True

        return False

    """ compute and return distance, direction and angle matrix """
    def retpreComputeStates(self, fishes):
        ## create containers for location, swimming directions, and 
        locations     = np.empty(shape=(self.N, self.dim ), dtype=float)
        curDirections = np.empty(shape=(self.N, self.dim ), dtype=float)
        cutOff        = np.empty(shape=(self.N, ),  dtype=float)

        ## fill matrix with locations / current swimming direction
        for i,fish in enumerate(fishes):
            locations[i,:]     = fish.location
            curDirections[i,:] = fish.curDirection
            cutOff[i]          = fish.sigmaPotential

        """
        locations[0,:]     = [0, 0]
        locations[1,:]     = [1, 0]
        curDirections[0,:] = [1, 0]
        curDirections[1,:] = [0, 1]
        """

        # normalize swimming directions NOTE the directions should already be normalized
        normalCurDirections = curDirections/(np.sqrt(np.einsum('ij,ij->i', curDirections, curDirections))[:,np.newaxis])

        ## create containers for direction, distance, and angle
        directionsOtherFish    = np.empty(shape=(self.N,self.N, self.dim ), dtype=float)
        distances              = np.empty(shape=(self.N,self.N),    dtype=float)
        anglesVel              = np.empty(shape=(self.N,self.N),    dtype=float)

        ## use numpy broadcasting to compute direction, distance, and angles
        directionsOtherFish         = locations[np.newaxis, :, :] - locations[:, np.newaxis, :]
        distances                   = np.sqrt( np.einsum('ijk,ijk->ij', directionsOtherFish, directionsOtherFish) )
        normalDirectionsOtherFish   = directionsOtherFish / distances[:,:,np.newaxis]
        # Filling diagonal to avoid division by 0
        np.fill_diagonal( distances, 1.0 )
        
        angles = None
        anglesPhi, anglesTheta = None, None
        anglesvPhi, anglesvTheta = None, None

        if self.dim == 2:
            
            acd = np.arctan2(normalCurDirections[:,1], normalCurDirections[:,0])
            anglesPhi = np.arctan2(normalDirectionsOtherFish[:,:,1], normalDirectionsOtherFish[:,:,0]) - acd[:,np.newaxis]
            anglesPhi = anglesPhi % (2*np.pi)
            anglesPhi[anglesPhi>np.pi] -= 2*np.pi
            anglesPhi[anglesPhi<-np.pi] += 2*np.pi
             
            anglesvPhi = acd[np.newaxis,:].T - acd[np.newaxis,:]
            anglesvPhi = anglesvPhi % (2*np.pi)
            anglesvPhi[anglesPhi>np.pi] -= 2*np.pi
            anglesvPhi[anglesPhi<-np.pi] += 2*np.pi
            
            # angle between the two vectors
            angles = anglesPhi

        else:
 
            # project on 2d
            curDirectionsXY = normalCurDirections[:,:-1]
            normalDirectionsOtherFishXY = normalDirectionsOtherFish[:,:,:-1]

            acd = np.arctan2(curDirectionsXY[:,1], curDirectionsXY[:,0])
            anglesPhi = np.arctan2(normalDirectionsOtherFishXY[:,:,1], normalDirectionsOtherFish[:,:,0]) - acd[:,np.newaxis]
            anglesPhi = anglesPhi % (2*np.pi)
            anglesPhi[anglesPhi>np.pi] -= 2*np.pi
            anglesPhi[anglesPhi<-np.pi] += 2*np.pi
            
            anglesvPhi = acd[np.newaxis,:].T - acd[np.newaxis,:]
            anglesvPhi = anglesPhi % (2*np.pi)
            anglesvPhi[anglesPhi>np.pi] -= 2*np.pi
            anglesvPhi[anglesPhi<-np.pi] += 2*np.pi
 
            acdz = np.arccos(curDirections[:,-1])
            anglesTheta = np.arccos(normalDirectionsOtherFish[:,:,-1]) - np.arccos(curDirections[:,-1])
            #anglesTheta = anglesTheta % (2*np.pi)

            anglesvTheta = acdz[np.newaxis,:] - acdz[:,np.newaxis]
            #anglesvTheta = anglesvTheta % (2*np.pi)

            np.fill_diagonal( anglesTheta, 0.)

            assert (np.abs(anglesTheta) <= np.pi).all(), f"[swarm] illegal theta state"
            assert (np.abs(anglesvTheta) <= np.pi).all(), f"[swarm] illegal vtheta state"
            
            # angle between the two vectors
            angles = np.arccos(np.einsum( 'ijk, ijk->ij', normalCurDirections[:,np.newaxis,:], normalDirectionsOtherFish ))
            np.fill_diagonal( anglesTheta, 0.)

        np.fill_diagonal( distances, np.inf )
        np.fill_diagonal( anglesPhi, 0.)

        return directionsOtherFish, distances, angles, anglesPhi, anglesTheta, anglesvPhi, anglesvTheta, cutOff

    """ compute distance and angle matrix """
    def preComputeStates(self):
        ## fill values to class member variable
        self.directionMat,  self.distancesMat, self.anglesMat, self.anglesPhiMat, self.anglesThetaMat, self.anglesvPhiMat, self.anglesvThetaMat, cutOff = self.retpreComputeStates(self.fishes)
        return False 

    def getState( self, i ):
        visible    = np.full(self.N, True)
        visible[i] = False # we cannot see outself
        
        distances  = self.distancesMat[i,visible]

        # sort and select nearest neighbours
        idSorted = np.argsort( distances )
        idNearestNeighbours = idSorted[:self.numNearestNeighbours]

        kernelDistancesNearestNeighbours = 1./np.sqrt(2.*np.pi*(0.33*self.rAttraction)**2)*np.exp( - 0.5 * (distances[idNearestNeighbours]/(self.rAttraction*0.33))**2 )

        if self.dim == 2:
        	anglesPhi = self.anglesPhiMat[i,visible]
        	anglesPhiNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesPhiNearestNeighbours[:self.numNearestNeighbours] = anglesPhi[idNearestNeighbours]

        	anglesvPhi = self.anglesvPhiMat[i,visible]
        	anglesvPhiNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesvPhiNearestNeighbours[:self.numNearestNeighbours] = anglesvPhi[idNearestNeighbours]

        	return np.array([kernelDistancesNearestNeighbours, anglesPhiNearestNeighbours, anglesvPhiNearestNeighbours]).flatten()
        else:
        	anglesPhi = self.anglesPhiMat[i,visible]
        	anglesPhiNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesPhiNearestNeighbours[:self.numNearestNeighbours] = anglesPhi[idNearestNeighbours]
        	
        	anglesTheta = self.anglesThetaMat[i,visible]
        	anglesThetaNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesThetaNearestNeighbours[:self.numNearestNeighbours] = anglesTheta[idNearestNeighbours]

        	anglesvPhi = self.anglesvPhiMat[i,visible]
        	anglesvPhiNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesvPhiNearestNeighbours[:self.numNearestNeighbours] = anglesvPhi[idNearestNeighbours]

        	anglesvTheta = self.anglesvThetaMat[i,visible]
        	anglesvThetaNearestNeighbours = np.full(self.numNearestNeighbours, -np.pi)
        	anglesvThetaNearestNeighbours[:self.numNearestNeighbours] = anglesvTheta[idNearestNeighbours]

        	return np.array([kernelDistancesNearestNeighbours, anglesPhiNearestNeighbours, anglesThetaNearestNeighbours, anglesvPhiNearestNeighbours, anglesvThetaNearestNeighbours]).flatten()
        
 
    def getGlobalReward( self ):
        # Careful: assumes sim.getState(i) was called before
        angMom = self.computeAngularMom()
        return np.full( self.N, angMom )

    def getLocalReward( self ):
        # Careful: assumes sim.getState(i) was called before
        center = self.computeCenter()
        returnVec = np.zeros(shape=(self.N,), dtype=float)
        
        if(self.dim == 2):
            #in this case the cross product yields a scalar
            angularMomentumVecSingle = np.zeros(shape=(self.N,), dtype=float)
            angularMomentumVec = 0.
            for i, fish in enumerate(self.fishes):
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle[i] = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle[i]
            
            signAngularMomentumVec = np.sign(angularMomentumVec)
            for i in range(self.N):
                returnVec[i] = angularMomentumVecSingle[i] * signAngularMomentumVec
 
        elif(self.dim == 3):
            angularMomentumVecSingle = np.zeros(shape=(self.N, self.dim), dtype=float)
            angularMomentumVec = np.zeros(shape=(self.dim,), dtype=float)
            for i, fish in enumerate(self.fishes):
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle[i,:] = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle[i,:]

            normAngularMomentumVec = np.linalg.norm(angularMomentumVec)
            unitAngularMomentumVec = angularMomentumVec / normAngularMomentumVec

            for i in range(self.N):
                returnVec[i] = np.dot(angularMomentumVecSingle[i,:], unitAngularMomentumVec)
       
        return returnVec

    """for fish i returns the repell, orient and attractTargets"""
    def retturnrep_or_att(self, i, fish, anglesMat, distancesMat):
        deviation = anglesMat[i,:]
        distances = distancesMat[i,:]
        visible = abs(deviation) <= ( self.alpha / 2. ) # check if the angle is within the visible range alpha

        rRepell  = self.rRepulsion   * ( 1 + fish.epsRepell  )
        rOrient  = self.rOrientation * ( 1 + fish.epsOrient  )
        rAttract = self.rAttraction  * ( 1 + fish.epsAttract )

        repellTargets  = self.fishes[(distances < rRepell)]
        orientTargets  = self.fishes[(distances >= rRepell) & (distances < rOrient) & visible]
        attractTargets = self.fishes[(distances >= rOrient) & (distances <= rAttract) & visible]

        return repellTargets, orientTargets, attractTargets


    # Careful assumes that precomputestates has already been called.
    ''' according to https://doi.org/10.1006/jtbi.2002.3065 and/or https://hal.archives-ouvertes.fr/hal-00167590 '''
    def move_calc(self):
        for i,fish in enumerate(self.fishes):
            repellTargets, orientTargets, attractTargets = self.retturnrep_or_att(i, fish, self.anglesMat, self.distancesMat)
            self.fishes[i].computeDirection(repellTargets, orientTargets, attractTargets, self.nu)


    ''' utility to compute polarisation (~alignement) '''
    def computePolarisation(self):
        polarisationVec = np.zeros(shape=(self.dim,), dtype=float)
        for fish in self.fishes:
            polarisationVec += fish.curDirection
        polarisation = np.linalg.norm(polarisationVec) / self.N
        return polarisation

    ''' utility to compute center of swarm '''
    def computeCenter(self):
        center = np.zeros(shape=(self.dim,), dtype=float)
        for fish in self.fishes:
            center += fish.location
        center /= self.N
        return center

    '''utility to compute average distance to center of the fishes'''
    def computeAvgDistCenter(self, center):
        # center = self.computeCenter()
        avg = 0.
        for fish in self.fishes:
            avg += np.linalg.norm(fish.location - center)
        avg /= self.N
        return avg


    ''' utility to compute angular momentum (~rotation) '''
    def computeAngularMom(self):
        center = self.computeCenter()

        if(self.dim == 2):
            #in this case the cross product yields a scalar
            angularMomentumVec = np.zeros(shape=(1,), dtype=float)
            for fish in self.fishes:
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle
            angularMomentum = np.linalg.norm(angularMomentumVec) / self.N

        elif(self.dim == 3):
            angularMomentumVec = np.zeros(shape=(self.dim,), dtype=float)
            for fish in self.fishes:
                distance = fish.location-center
                distanceNormal = distance / np.linalg.norm(distance) 
                angularMomentumVecSingle = np.cross(distanceNormal,fish.curDirection)
                angularMomentumVec += angularMomentumVecSingle
            angularMomentum = np.linalg.norm(angularMomentumVec) / self.N

        return angularMomentum

    def printstate(self):
        # N, numNN, numdimensions, movementType, initType, _psi, seed=43, _rRepulsion = 0.1, _delrOrientation=1.5, _delrAttraction=3, _alpha=1.5*np.pi, _initcircle = 1.
        print("N :", self.N)
        print("numNN :", self.numNearestNeighbours)
        print("numdimensions :", self.dim)
        print("initType :", self.initializationType)
        print("psi :", self.psi)
        print("seed :", self.seed)
        print("rRepulsion :", self.rRepulsion)
        print("rOrientation :", self.rOrientation)
        print("rAttraction :", self.rAttraction)
        print("alpha :", self.alpha)
        print("initcircle :", self.initialCircle)
