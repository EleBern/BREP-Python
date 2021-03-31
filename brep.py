# BREP-Python
# Author: Ohki Katakura (University of Hertfordshire)
# Contact: contact@neuronalpail.com

# This is translation of BREP by Ivan Raikov.
# This script require coordination of GrCs, TJs and GoCs
# and generate coordination of AAs, PFs, ADs, BDs, and GoC axons
# and connectivity of AAs to GoCs (ADs/BDs), PFs to GoCs (ADs),
# GoCs to GoCs for inhibitory synapses (axons to soma) and
# gap junctions (soma to soma).

###########################################################
### Licence, GNU General Public License v3.0 (GPL3)     ###
###########################################################

# Copyright 2021 Ohki Katakura

# BREP-Python is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This programme is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this programme. If not, see <https://www.gnu.org/licenses/>.


###########################################################
### Libraries                                           ###
###########################################################

import os,sys
import numpy as np
from neuron import h
from mpi4py import MPI
from scipy.spatial import KDTree
import time
from argparse import ArgumentParser


###########################################################
### System variables                                    ###
###########################################################

tb0 = time.time()

MPIcomm = MPI.COMM_WORLD
MPIsize = MPIcomm.Get_size()
MPIrank = MPIcomm.Get_rank()


###########################################################
### Parameters                                          ###
### Most of them are put in Parameters.hoc              ###
###########################################################

GoCAxonSegs = 1
GoCAxonPts = 2
K4T = 1000 # K value for KDTree; enough large to catch all neighbouring nodes
np.random.seed(113)


###########################################################
### Arguments/Commandline options                       ###
###########################################################

usage = 'mpiexec -n <NUM THREADS> python3 {0} [-hlvt] [-r <RANDOM TABLE FILE>]'.format(__file__)
argparser = ArgumentParser(usage=usage)
argparser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='show all messages. Additional messages are shown with (v).')
argparser.add_argument('-l', '--loadFiles',
                        action='store_true',
                        help='Load existing coordinates files generated BREP.')
argparser.add_argument('-t', '--testMode',
                        action='store_true',
                        help='Set TEST MODE: Stop when the coordinates are created.')
argparser.add_argument('-c', '--chunk',
                        action='store_true',
                        help='<not implemented>')
argparser.add_argument('-r', '--randomTable',
                        action='store',
                        default=None,
                        type=str,
                        help='File of random number table (0,1) instead of numpy.random.normal. Default: None.')
args = argparser.parse_args()

prmdir=os.getenv('PARAMDIR')
h.xopen(os.path.join(prmdir, 'Parameters.hoc'))

def vprint(s):
    if args.verbose:
        print('(v)', s)


print('brep.py rank {0} args.verbose: {0}'.format(MPIrank, args.verbose))

###########################################################
### Load/generate coordinates                           ###
###########################################################

tb = time.time()
if args.loadFiles:
    vprint('brep.py rank {0} loadFiles: {1}'.format(MPIrank, args.loadFiles))
    GrC = np.loadtxt('GCcoordinates.sorted.dat')
    TJ  = np.loadtxt('GCTcoordinates.sorted.dat')
    GoC = np.loadtxt('GoCcoordinates.sorted.dat')

    AAs = np.loadtxt('AAcoordinates{0}.dat'.format(MPIrank))
    AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T

    PFs = np.loadtxt('PFcoordinates{0}.dat'.format(MPIrank))
    PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T

    GoCadend = np.loadtxt('GoCadendcoordinates.sorted.dat')
    adendPerGoC = GoCadend.shape[1]//3
    GoCadend = GoCadend.reshape(GoCadend.shape[0]*(GoCadend.shape[1]//3), 3)
    GoCadendIdx = np.arange(len(GoCadend))//adendPerGoC
    adendSegIdx = np.arange(len(GoCadend))%adendPerGoC
    adendSecIdx = adendSegIdx%(h.GoC_Ad_nseg*h.GoC_Ad_nsegpts)//(h.GoC_Ad_nsegpts)+1
    adendDendIdx = h.numDendGolgi - adendSegIdx//(h.GoC_Ad_nseg*h.GoC_Ad_nsegpts)

    GoCbdend = np.loadtxt('GoCbdendcoordinates.sorted.dat')
    bdendPerGoC = GoCbdend.shape[1]//3
    GoCbdend = GoCbdend.reshape(GoCbdend.shape[0]*(GoCbdend.shape[1]//3), 3)
    GoCbdendIdx = np.arange(len(GoCbdend))//bdendPerGoC
    bdendSegIdx = np.arange(len(GoCbdend))%bdendPerGoC
    bdendSecIdx = bdendSegIdx%(h.GoC_Bd_nseg*h.GoC_Bd_nsegpts)//(h.GoC_Bd_nsegpts)+1
    bdendDendIdx = h.numDendGolgi/2 - bdendSegIdx//(h.GoC_Bd_nseg*h.GoC_Bd_nsegpts)

    GoCdend = np.vstack([GoCadend, GoCbdend])
    GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
    dendSegIdx = np.hstack([adendSegIdx, bdendSegIdx])
    dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
    dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])

    GoCaxon = np.loadtxt('GoCaxoncoordinates.sorted.dat')
    axonPerGoC = GoCaxon.shape[1]//3
    GoCaxon = GoCaxon.reshape(GoCaxon.shape[0]*(GoCaxon.shape[1]//3), 3)
    GoCaxonIdx = np.arange(len(GoCaxon))//axonPerGoC
    axonSegIdx = np.arange(len(GoCaxon))%axonPerGoC
    axonSecIdx = axonSegIdx%(GoCAxonSegs*GoCAxonPts)//(GoCAxonPts)+1
    axonDendIdx = h.numAxonGolgi - axonSegIdx//(GoCAxonSegs*GoCAxonPts)
    vprint('brep.py rank {0} Finish to load files ({1:.2f} s)'.format(MPIrank, time.time()-tb))

else:
    vprint('brep.py rank {0} loadFiles: {1}'.format(MPIrank, args.loadFiles))
    if os.path.isfile('GCcoordinates.sorted.dat') and os.path.isfile('GCTcoordinates.sorted.dat'):
        GrC = np.loadtxt('GCcoordinates.sorted.dat')
        TJ = np.loadtxt('GCTcoordinates.sorted.dat')
    else:
        GrC = np.loadtxt('GCcoordinates.dat')
        np.savetxt('GCcoordinates.sorted.dat', GrC, '%g')
        TJ  = np.loadtxt('GCTcoordinates.dat')
        np.savetxt('GCTcoordinates.sorted.dat', TJ, '%g')
    if os.path.isfile('GoCcoordinates.sorted.dat'):
        GoC = np.loadtxt('GoCcoordinates.sorted.dat')
    else:
        GoC = np.loadtxt('GoCcoordinates.dat')
        np.savetxt('GoCcoordinates.sorted.dat', GoC, '%g')

    GrCidx = np.arange(MPIrank, len(GrC), MPIsize).astype(np.int64)

    # Ascending Axons (AAs)
    maxAAlen = h.GLdepth + h.PCLdepth
    AAlength = TJ[GrCidx,2] - GrC[GrCidx,2]
    # AAlength[AAlength>maxAAlen] = maxAAlen
    AAlength = maxAAlen*np.ones(len(GrCidx))
    numAAs = np.floor(AAlength / h.AAstep)
    AAs = np.stack(np.meshgrid(np.arange(numAAs.max()), AAlength / (numAAs-1))).prod(axis=0)
    AAs = np.vstack([GrCidx[np.arange(AAs.shape[0]*AAs.shape[1])//AAs.shape[1]], AAs.flatten()]).T
    AAs = AAs[AAs[:,1]<=maxAAlen]
    AAs = np.vstack([
        AAs[:,0],
        GrC[AAs[:,0].astype(np.int64),:2].T,
        GrC[AAs[:,0].astype(np.int64),2] + AAs[:,1]
    ]).T
    AAs = AAs[np.lexsort([AAs[:,3], AAs[:,2], AAs[:,1]])]
    if args.testMode:
        np.savetxt('pyAAcoordinates{0}.dat'.format(MPIrank), AAs, fmt='%g')
    else:
        np.savetxt('AAcoordinates{0}.dat'.format(MPIrank), AAs, fmt='%g')
    AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T

    # Parallel Fibres (PFs)
    numPFs = np.floor(h.PFlength/h.PFstep).astype(np.int64)
    PFs = np.linspace(-h.PFlength/2, h.PFlength/2, numPFs)
    PFs = np.tile(PFs,[len(GrCidx),1])
    PFs = np.vstack([GrCidx[np.arange(PFs.shape[0]*PFs.shape[1])//PFs.shape[1]], PFs.flatten()]).T
    PFs = np.vstack([
        PFs[:,0],
        TJ[PFs[:,0].astype(np.int64),0] + PFs[:,1],
        TJ[PFs[:,0].astype(np.int64),1:].T
    ]).T
    PFs = PFs[np.lexsort([PFs[:,3], PFs[:,2], PFs[:,1]])].astype(np.float64)
    if args.testMode:
        np.savetxt('pyPFcoordinates{0}.dat'.format(MPIrank), PFs, fmt='%g')
    else:
        np.savetxt('PFcoordinates{0}.dat'.format(MPIrank), PFs, fmt='%g')
    PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T

    # random numbers for GoC dendrites
    if args.randomTable:
        from itertools import cycle
        theta = cycle(np.loadtxt(args.randomTable))
        theta = [next(theta) for i in range(len(GoC)*int(h.numDendGolgi))]
        theta = np.array(theta).reshape([len(GoC), int(h.numDendGolgi)])
    else:
        np.random.seed(73)
        theta = np.random.normal(0,1,[len(GoC), int(h.numDendGolgi)])
    thetaStd = np.array([h.GoC_Btheta_stdev, h.GoC_Btheta_stdev, h.GoC_Atheta_stdev, h.GoC_Atheta_stdev])
    thetaMean = np.array([h.GoC_Btheta_max, h.GoC_Btheta_min, h.GoC_Atheta_max, h.GoC_Atheta_min])
    theta = theta * thetaStd + thetaMean

    # Basolateral dendrites (BDs)
    GoCbdend = []
    nseg = int(h.GoC_Bd_nseg*h.GoC_Bd_nsegpts)
    target = np.vstack([
        h.GoC_PhysBasolateralDendR * np.cos(theta[:,0]*np.pi/180),
        h.GoC_PhysBasolateralDendR * np.sin(theta[:,0]*np.pi/180),
        h.GoC_PhysBasolateralDendH * np.ones(len(theta))
    ]).T
    GoCbdend.append(np.linspace(GoC, GoC+target, nseg).transpose(1,0,2))
    target = np.vstack([
        h.GoC_PhysBasolateralDendR*np.cos(theta[:,1]*np.pi/180),
        h.GoC_PhysBasolateralDendR*np.sin(theta[:,1]*np.pi/180),
        h.GoC_PhysBasolateralDendH*np.ones(len(theta))
    ]).T
    GoCbdend.append(np.linspace(GoC, GoC+target, nseg).transpose(1,0,2))
    GoCbdend = np.hstack(GoCbdend).reshape(len(GoC), nseg*2*3)
    del(target)
    if MPIrank == 0:
        if args.testMode:
            np.savetxt('pyGoCbdendcoordinates.sorted.dat', GoCbdend, fmt='%g')
        else:
            np.savetxt('GoCbdendcoordinates.sorted.dat', GoCbdend, fmt='%g')
    bdendPerGoC = GoCbdend.shape[1]//3
    GoCbdend = GoCbdend.reshape(GoCbdend.shape[0]*(GoCbdend.shape[1]//3), 3)
    GoCbdendIdx = np.arange(len(GoCbdend))//bdendPerGoC
    bdendSegIdx = np.arange(len(GoCbdend))%bdendPerGoC
    bdendSecIdx = bdendSegIdx%(h.GoC_Bd_nseg*h.GoC_Bd_nsegpts)//(h.GoC_Bd_nsegpts)+1
    bdendDendIdx = h.numDendGolgi/2 - bdendSegIdx//(h.GoC_Bd_nseg*h.GoC_Bd_nsegpts)

    # Apical dendrites (ADs)
    GoCadend = []
    nseg = int(h.GoC_Ad_nseg*h.GoC_Ad_nsegpts)
    target = np.vstack([
        h.GoC_PhysApicalDendR*np.cos(theta[:,2]*np.pi/180),
        h.GoC_PhysApicalDendR*np.sin(theta[:,2]*np.pi/180),
        h.GoC_PhysApicalDendH*np.ones(len(theta))
    ]).T
    GoCadend.append(np.linspace(GoC, GoC+target, nseg).transpose(1,0,2))
    target = np.vstack([
        h.GoC_PhysApicalDendR*np.cos(theta[:,3]*np.pi/180),
        h.GoC_PhysApicalDendR*np.sin(theta[:,3]*np.pi/180),
        h.GoC_PhysApicalDendH*np.ones(len(theta))
    ]).T
    GoCadend.append(np.linspace(GoC, GoC+target, nseg).transpose(1,0,2))
    GoCadend = np.hstack(GoCadend).reshape(len(GoC), nseg*2*3)
    del(target)
    if MPIrank == 0:
        if args.testMode:
            np.savetxt('pyGoCadendcoordinates.sorted.dat', GoCadend, fmt='%g')
        else:
            np.savetxt('GoCadendcoordinates.sorted.dat', GoCadend, fmt='%g')
    adendPerGoC = GoCadend.shape[1]//3
    GoCadend = GoCadend.reshape(GoCadend.shape[0]*(GoCadend.shape[1]//3), 3)
    GoCadendIdx = np.arange(len(GoCadend))//adendPerGoC
    adendSegIdx = np.arange(len(GoCadend))%adendPerGoC
    adendSecIdx = adendSegIdx%(h.GoC_Ad_nseg*h.GoC_Ad_nsegpts)//(h.GoC_Ad_nsegpts)+1
    adendDendIdx = h.numDendGolgi - adendSegIdx//(h.GoC_Ad_nseg*h.GoC_Ad_nsegpts)

    # dendrites
    GoCdend = np.vstack([GoCadend, GoCbdend])
    GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
    dendSegIdx = np.hstack([adendSegIdx, bdendSegIdx])
    dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
    dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])

    # GoC axons
    np.random.seed(79)
    GoCaxon = np.random.random([len(GoC), int(h.numAxonGolgi), 3])
    GoCaxon[:,:,0] = h.GoC_Axon_Xmin + np.floor(
                                (h.GoC_Axon_Xmax - h.GoC_Axon_Xmin + 1)*GoCaxon[:,:,0])
    GoCaxon[:,:,1] = h.GoC_Axon_Ymin + np.floor(
                                (h.GoC_Axon_Ymax - h.GoC_Axon_Ymin + 1)*GoCaxon[:,:,1])
    GoCaxon[:,:,2] = h.GoC_Axon_Zmin + np.floor(
                                (h.GoC_Axon_Zmax - h.GoC_Axon_Zmin + 1)*GoCaxon[:,:,2])
    g = np.tile(GoC,int(h.numAxonGolgi)).reshape(19,20,3)
    GoCaxon = np.stack([g, g+GoCaxon], axis=2).reshape(len(GoC), int(h.numAxonGolgi*2*3))
    del(g)
    if MPIrank == 0:
        if args.testMode:
            np.savetxt('pyGoCaxoncoordinates.sorted.dat', GoCaxon, fmt='%g')
        else:
            np.savetxt('GoCaxoncoordinates.sorted.dat', GoCaxon, fmt='%g')
    axonPerGoC = GoCaxon.shape[1]//3
    GoCaxon = GoCaxon.reshape(GoCaxon.shape[0]*(GoCaxon.shape[1]//3), 3)
    GoCaxonIdx = np.arange(len(GoCaxon))//axonPerGoC
    axonSegIdx = np.arange(len(GoCaxon))%axonPerGoC
    axonSecIdx = axonSegIdx%(GoCAxonSegs*GoCAxonPts)//(GoCAxonPts)+1
    axonDendIdx = h.numAxonGolgi - axonSegIdx//(GoCAxonSegs*GoCAxonPts)

    vprint('brep.py rank {0} Finish to create coordinations ({1:.2f} s)'.format(MPIrank, time.time()-tb))

if args.testMode:
    print('TEST MODE')
    sys.exit(1)


###########################################################
### GrC PFs to GoC AD for glutamatergic synapses     ###
###########################################################

vprint('brep.py rank {0} PFs to GoCs'.format(MPIrank))
tb = time.time()
tree = KDTree(PFs[:,2:])
K = min(len(PFs), K4T)
distKDTree,idxKDTree = tree.query(GoCadend, k=K)

results = np.stack([
    np.repeat(np.arange(len(GoCadend)), K).reshape(len(GoCadend), K),
    PFs[idxKDTree,1],
    distKDTree,
    idxKDTree,
]).transpose(1,2,0)
results = results.reshape(results.shape[0]*results.shape[1], results.shape[2])
results = results[results[:,2]<=h.PFtoGoCzone]
results = results[np.unique(results[:,:2], axis=0, return_index=True)[1]]
dendLen = np.linalg.norm(GoCadend[results[:,0].astype(np.int64)]
                            - GoC[GoCadendIdx[results[:,0].astype(np.int64)]], axis=1)
axonLen = np.linalg.norm(PFs[results[:,3].astype(np.int64),2:]
                            - TJ[results[:,1].astype(np.int64)], axis=1)
results = np.vstack([
    results[:,1],
    GoCadendIdx[results[:,0].astype(np.int64)],
    results[:,2] + dendLen + axonLen,
    adendSecIdx[results[:,0].astype(np.int64)],
    adendDendIdx[results[:,0].astype(np.int64)]
]).T
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:,1]%MPIsize==MPIrank]
np.savetxt('PFtoGoCsources{0}.dat'.format(MPIrank), results[:,0], fmt='%d')
np.savetxt('PFtoGoCtargets{0}.dat'.format(MPIrank), results[:,1], fmt='%d')
np.savetxt('PFtoGoCdistances{0}.dat'.format(MPIrank), results[:,2], fmt='%f')
np.savetxt('PFtoGoCsegments{0}.dat'.format(MPIrank), results[:,[3,4]], fmt='%d')
vprint('brep.py rank {0} PFs to GoCs ({1:.2f} s)'.format(MPIrank, time.time()-tb))


###########################################################
### GrC AAs to GoC AD/BD for glutamatergic synapses     ###
###########################################################

vprint('brep.py rank {0} AAs to GoCs'.format(MPIrank))
tree = KDTree(AAs[:,2:])
K = min(len(AAs), K4T)
distKDTree,idxKDTree = tree.query(GoCdend, k=K)

results = np.stack([
    np.repeat(np.arange(len(GoCdend)), K).reshape(len(GoCdend), K),
    AAs[idxKDTree,1],
    distKDTree,
    idxKDTree,
]).transpose(1,2,0)
results = results.reshape(results.shape[0]*results.shape[1], results.shape[2])
results = results[results[:,2]<=h.AAtoGoCzone]
results = results[np.unique(results[:,:2], axis=0, return_index=True)[1]]
dendLen = np.linalg.norm(GoCdend[results[:,0].astype(np.int64)]
                            - GoC[GoCdendIdx[results[:,0].astype(np.int64)]], axis=1)
axonLen = np.linalg.norm(AAs[results[:,3].astype(np.int64),2:]
                            - GrC[results[:,1].astype(np.int64)], axis=1)
results = np.vstack([
    results[:,1],
    GoCdendIdx[results[:,0].astype(np.int64)],
    results[:,2] + dendLen + axonLen,
    dendSecIdx[results[:,0].astype(np.int64)],
    dendDendIdx[results[:,0].astype(np.int64)]
]).T
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:,1]%MPIsize==MPIrank]
np.savetxt('AAtoGoCsources{0}.dat'.format(MPIrank), results[:,0], fmt='%d')
np.savetxt('AAtoGoCtargets{0}.dat'.format(MPIrank), results[:,1], fmt='%d')
np.savetxt('AAtoGoCdistances{0}.dat'.format(MPIrank), results[:,2], fmt='%f')
np.savetxt('AAtoGoCsegments{0}.dat'.format(MPIrank), results[:,[3,4]], fmt='%d')
vprint('brep.py rank {0} AAs to GoCs ({1:.2f} s)'.format(MPIrank, time.time()-tb))

###########################################################
### GoC axon to GoC soma for GABAergic synapses         ###
###########################################################

vprint('brep.py rank {0} GoCs to GoCs inh'.format(MPIrank))
idx = np.where(GoCaxonIdx%MPIsize==MPIrank)[0]
axonForTree = np.vstack([np.arange(len(idx)), GoCaxonIdx[idx], GoCaxon[idx].T]).T

tree = KDTree(axonForTree[:,2:])
K = min(len(axonForTree), K4T)
distKDTree,idxKDTree = tree.query(GoC, k=K)

results = np.stack([
    np.repeat(np.arange(len(GoC)), K).reshape(len(GoC), K),
    axonForTree[idxKDTree,1],
    distKDTree,
    idxKDTree,
]).transpose(1,2,0)
results = results.reshape(results.shape[0]*results.shape[1], results.shape[2])
results = results[(results[:,2]<=h.GoCtoGoCzone) & (results[:,0] != results[:,1])
        & (GoC[results[:,0].astype(np.int64),2] < axonForTree[results[:,3].astype(np.int64),4])]
results = results[np.unique(results[:,:2], axis=0, return_index=True)[1]]
dendLen = 0 # soma to soma
axonLen = np.linalg.norm(axonForTree[results[:,3].astype(np.int64),2:]
                            - GoC[results[:,1].astype(np.int64)], axis=1)
results = np.vstack([
    results[:,1],
    results[:,0],
    results[:,2] + dendLen + axonLen
]).T
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:,1]%MPIsize==MPIrank]
np.savetxt('GoCtoGoCsources{0}.dat'.format(MPIrank), results[:,0], fmt='%d')
np.savetxt('GoCtoGoCtargets{0}.dat'.format(MPIrank), results[:,1], fmt='%d')
np.savetxt('GoCtoGoCdistances{0}.dat'.format(MPIrank), results[:,2], fmt='%f')
vprint('brep.py rank {0} GoCs to GoCs inh ({1:.2f} s)'.format(MPIrank, time.time()-tb))



###########################################################
### GoC soma to GoC soma for gap junction               ###
###########################################################

vprint('brep.py rank {0} GoCs to GoCs gap'.format(MPIrank))
GoCidx = np.arange(len(GoC))
idx = np.where(GoCidx%MPIsize==MPIrank)[0]
GoCForTree = np.vstack([np.arange(len(idx)), GoCidx[idx], GoC[idx].T]).T

tree = KDTree(GoCForTree[:,2:])
K = min(len(GoCForTree), K4T)
distKDTree,idxKDTree = tree.query(GoC, k=K)

results = np.stack([
    np.repeat(np.arange(len(GoC)), K).reshape(len(GoC), K),
    GoCForTree[idxKDTree,1],
    distKDTree,
    idxKDTree,
]).transpose(1,2,0)
results = results.reshape(results.shape[0]*results.shape[1], results.shape[2])
results = results[(results[:,2]<=h.GoCtoGoCgapzone) & (results[:,0] != results[:,1])]
results = results[np.unique(results[:,:2], axis=0, return_index=True)[1]]
results = np.vstack([
    results[:,1],
    results[:,0],
    results[:,2]
]).T
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:,1]%MPIsize==MPIrank]
np.savetxt('GoCtoGoCgapsources{0}.dat'.format(MPIrank), results[:,0], fmt='%d')
np.savetxt('GoCtoGoCgaptargets{0}.dat'.format(MPIrank), results[:,1], fmt='%d')
np.savetxt('GoCtoGoCgapdistances{0}.dat'.format(MPIrank), results[:,2], fmt='%f')
vprint('brep.py rank {0} GoCs to GoCs gap ({1:.2f} s)'.format(MPIrank, time.time()-tb))


vprint('brep.py rank {0} completed ({1:.2f} s)'.format(MPIrank, time.time()-tb0))


