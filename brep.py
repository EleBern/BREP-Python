#!/usr/bin/env python
"""
BREP-Python
Author: Ohki Katakura (University of Hertfordshire)
Contact: contact@neuronalpail.com

This is translation of BREP by Ivan Raikov.
This script require coordination of GrCs, TJs and GoCs
and generate coordination of AAs, PFs, ADs, BDs, and GoC axons
and connectivity of AAs to GoCs (ADs/BDs), PFs to GoCs (ADs),
GoCs to GoCs for inhibitory synapses (axons to soma) and
gap junctions (soma to soma).

Update on August 2022: Configure orders to save memory space.

#######################################################
# Licence, GNU General Public License v3.0 (GPL3)     #
#######################################################

Copyright 2021-2022 Ohki Katakura

BREP-Python is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This programme is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this programme. If not, see <https://www.gnu.org/licenses/>.
"""


#######################################################
# Libraries                                           #
#######################################################

import os
import sys
import numpy as np
from neuron import h
from mpi4py import MPI
from scipy.spatial import KDTree
import time
from argparse import ArgumentParser
from itertools import cycle


#######################################################
# System variables                                    #
#######################################################

tb0 = time.time()

MPIcomm = MPI.COMM_WORLD
MPIsize = MPIcomm.Get_size()
MPIrank = MPIcomm.Get_rank()


#######################################################
# Parameters                                          #
# Most of them are put in Parameters.hoc              #
#######################################################

GoCAxonSegs = 1
GoCAxonPts = 2
K4T = 200  # K value for KDTree; enough large to catch all neighbouring nodes and enough small for memory capacity

GoCdendSeed = 73
GoCaxonSeed = 79


#######################################################
# Arguments/Command line options                      #
#######################################################

usage = f"mpiexec -n <NUM THREADS> python3 {__file__} [-hlvtapd] [-r <RANDOM TABLE FILE>]"
argparser = ArgumentParser(usage=usage)
argparser.add_argument("-v", "--verbose", action="store_true", help="show all messages. Additional messages are shown with (v).")
argparser.add_argument("-l", "--loadFiles", action="store_true", help="Load existing coordinates files generated BREP.")
argparser.add_argument("-m", "--memory", action="store_true", help="Sequentially execute KD Tree for PF-GoC to save memory space.")
argparser.add_argument("-t", "--testMode", action="store_true", help="Set TEST MODE: Stop when the coordinates are created.")
argparser.add_argument("-c", "--chunk", action="store_true", help="<not implemented>")
argparser.add_argument("-a", "--saveAA", action="store_true", help="Save AA coordinates (default: False)")
argparser.add_argument("-p", "--savePF", action="store_true", help="Save PF coordinates (default: False)")
argparser.add_argument("-d", "--saveDend", action="store_true", help="Save GoC dendrites and axon coordinates (default: False)")
argparser.add_argument(
    "-r",
    "--randomTable",
    action="store",
    default=None,
    type=str,
    help="Use file of random number instead of numpy.random.normal. Default: None.",
)
args = argparser.parse_args()

prmdir = os.getenv("PARAMDIR")
h.xopen(os.path.join(prmdir, "Parameters.hoc"))


def vprint(s, end="\n"):
    if args.verbose:
        print(f"(v) {s}", end=end)


def vload(s):
    vprint(f"brep.py rank {MPIrank}: load {s}")
    x = np.loadtxt(s)
    vprint(f"brep.py rank {MPIrank} done!")
    return x


def vsave(file, obj, fmt):
    vprint(f"brep.py rank {MPIrank}: save {file}")
    x = np.savetxt(file, obj, fmt=fmt)
    vprint(f"brep.py rank {MPIrank} done!")
    return x


if (
    os.path.isfile("GCcoordinates.sorted.dat")
    and os.path.isfile("GCTcoordinates.sorted.dat")
    and os.path.getsize("GCcoordinates.sorted.dat") > 0
    and os.path.getsize("GCTcoordinates.sorted.dat") > 0
):
    GrC = vload("GCcoordinates.sorted.dat")
    TJ = vload("GCTcoordinates.sorted.dat")
else:
    GrC = vload("GCcoordinates.dat")
    TJ = vload("GCTcoordinates.dat")

if os.path.isfile("GoCcoordinates.sorted.dat") and os.path.getsize("GoCcoordinates.sorted.dat") > 0:
    GoC = vload("GoCcoordinates.sorted.dat")
else:
    GoC = vload("GoCcoordinates.dat")
MPIcomm.barrier()
if MPIrank == 0:
    vsave("GCcoordinates.sorted.dat", GrC, "%g")
    vsave("GCTcoordinates.sorted.dat", TJ, "%g")
    vsave("GoCcoordinates.sorted.dat", GoC, "%g")
del GrC
del TJ
del GoC

print(f"brep.py rank {MPIrank} args.verbose: {args.verbose}")


####################################################
# GrC PFs to GoC AD for glutamatergic synapses     #
####################################################

tb = time.time()
if args.loadFiles:
    GrC = vload("GCcoordinates.sorted.dat")
    TJ = vload("GCTcoordinates.sorted.dat")

    GoC = vload("GoCcoordinates.sorted.dat")

    PFs = vload(f"PFcoordinates{MPIrank}.dat")
    PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T

    GoCadend = vload("GoCadendcoordinates.sorted.dat")
    adendPerGoC = GoCadend.shape[1] // 3
    GoCadend = GoCadend.reshape(GoCadend.shape[0] * (GoCadend.shape[1] // 3), 3)
    GoCadendIdx = np.arange(len(GoCadend)) // adendPerGoC
    adendSegIdx = np.arange(len(GoCadend)) % adendPerGoC
    adendSecIdx = adendSegIdx % (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts) // (h.GoC_Ad_nsegpts) + 1
    adendDendIdx = h.numDendGolgi - adendSegIdx // (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)
    del adendSegIdx

else:
    if (
        os.path.isfile("GCcoordinates.sorted.dat")
        and os.path.isfile("GCTcoordinates.sorted.dat")
        and os.path.getsize("GCcoordinates.sorted.dat") > 0
        and os.path.getsize("GCTcoordinates.sorted.dat") > 0
    ):
        GrC = vload("GCcoordinates.sorted.dat")
        TJ = vload("GCTcoordinates.sorted.dat")
    else:
        GrC = vload("GCcoordinates.dat")
        TJ = vload("GCTcoordinates.dat")

    if os.path.isfile("GoCcoordinates.sorted.dat") and os.path.getsize("GoCcoordinates.sorted.dat") > 0:
        GoC = vload("GoCcoordinates.sorted.dat")
    else:
        GoC = vload("GoCcoordinates.dat")

    if hasattr(h, "BREP_PFcoeff") and h.BREP_PFcoeff > 0:
        numPFs = np.floor(2 * h.PFlength * h.BREP_PFcoeff / h.PFstep).astype(np.int32)
        PFs = np.linspace(-h.PFlength * h.BREP_PFcoeff, h.PFlength * h.BREP_PFcoeff, numPFs).astype(np.float32)
    else:
        numPFs = np.floor(2 * h.PFlength / h.PFstep).astype(np.int32)
        PFs = np.linspace(-h.PFlength, h.PFlength, numPFs).astype(np.float32)
    del numPFs
    GrCidx = np.arange(MPIrank, len(TJ), MPIsize).astype(np.int32)
    PFs = np.tile(PFs, [len(GrCidx), 1]).astype(np.float32)
    PFs = np.vstack([GrCidx[np.arange(PFs.shape[0] * PFs.shape[1]) // PFs.shape[1]], PFs.flatten()]).T.astype(np.float32)
    del GrCidx
    PFs = np.vstack([PFs[:, 0], TJ[PFs[:, 0].astype(np.int32), 0] + PFs[:, 1], TJ[PFs[:, 0].astype(np.int32), 1:].T]).T.astype(np.float32)
    PFs = PFs[np.lexsort([PFs[:, 3], PFs[:, 2], PFs[:, 1]])].astype(np.float32)
    if args.savePF:
        if args.testMode:
            vsave(f"pyPFcoordinates{MPIrank}.dat", PFs, fmt="%g")
        else:
            vsave(f"PFcoordinates{MPIrank}.dat", PFs, fmt="%g")
    PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T.astype(np.float32)

    GoCadend = []
    nseg = int(h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)
    if args.randomTable:
        theta = cycle(vload(args.randomTable))
        theta = [next(theta) for i in range(len(GoC) * int(h.numDendGolgi))]
        theta = np.array(theta).reshape([len(GoC), int(h.numDendGolgi)])
    else:
        np.random.seed(GoCdendSeed)
        theta = np.random.normal(0, 1, [len(GoC), int(h.numDendGolgi)])
    thetaStd = np.array([h.GoC_Btheta_stdev, h.GoC_Btheta_stdev, h.GoC_Atheta_stdev, h.GoC_Atheta_stdev])
    thetaMean = np.array([h.GoC_Btheta_max, h.GoC_Btheta_min, h.GoC_Atheta_max, h.GoC_Atheta_min])
    theta = theta * thetaStd + thetaMean
    del thetaMean
    del thetaStd
    target = np.vstack(
        [
            h.GoC_PhysApicalDendR * np.cos(theta[:, 2] * np.pi / 180),
            h.GoC_PhysApicalDendR * np.sin(theta[:, 2] * np.pi / 180),
            h.GoC_PhysApicalDendH * np.ones(len(theta)),
        ]
    ).T
    GoCadend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
    target = np.vstack(
        [
            h.GoC_PhysApicalDendR * np.cos(theta[:, 3] * np.pi / 180),
            h.GoC_PhysApicalDendR * np.sin(theta[:, 3] * np.pi / 180),
            h.GoC_PhysApicalDendH * np.ones(len(theta)),
        ]
    ).T
    GoCadend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
    del target
    GoCadend = np.hstack(GoCadend).reshape(len(GoC), nseg * 2 * 3)
    del GoC  # take more time, save more memory space
    if args.saveDend:
        if MPIrank == 0:
            if args.testMode:
                vsave("pyGoCadendcoordinates.sorted.dat", GoCadend, fmt="%g")
            else:
                vsave("GoCadendcoordinates.sorted.dat", GoCadend, fmt="%g")
    adendPerGoC = GoCadend.shape[1] // 3
    GoCadend = GoCadend.reshape(GoCadend.shape[0] * (GoCadend.shape[1] // 3), 3)
    GoCadendIdx = np.arange(len(GoCadend)) // adendPerGoC
    adendSegIdx = np.arange(len(GoCadend)) % adendPerGoC
    adendSecIdx = adendSegIdx % (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts) // (h.GoC_Ad_nsegpts) + 1
    adendDendIdx = h.numDendGolgi - adendSegIdx // (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)
    del adendSegIdx

vprint(f"brep.py rank {MPIrank} Preparation for PFs to GoCs ({time.time() - tb:.2f} s)")


vprint(f"brep.py rank {MPIrank} PFs to GoCs")

tb = time.time()

if args.memory:
    for i in range(MPIsize):
        MPIcomm.barrier()
        if i == MPIrank:
            tree = KDTree(PFs[:, 2:])
            print(f"brep.py rank {MPIrank} KDtree built")
            K = min(len(PFs), K4T)
            distKDTree, idxKDTree = tree.query(GoCadend, k=K)
            print(f"brep.py rank {MPIrank} KDTree queried")
            del tree

            results = np.stack(
                [
                    np.repeat(np.arange(len(GoCadend)), K).reshape(len(GoCadend), K),
                    PFs[idxKDTree, 1],
                    distKDTree,
                    idxKDTree,
                ]
            ).transpose(1, 2, 0)
            del distKDTree
            del idxKDTree
            results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
            if results[:, 2].max() <= h.PFtoGoCzone:
                print(f"brep.py {MPIrank} all of {K} queries are within the zone")
            results = results[results[:, 2] <= h.PFtoGoCzone]
            results = results[np.unique(results[:, :2], axis=0, return_index=True)[1]]
            if hasattr(h, "BREP_dendcoeff") and h.BREP_dendcoeff > 0:
                dendLen = h.BREP_dendcoeff * np.linalg.norm(
                    GoCadend[results[:, 0].astype(np.int32)] - GoC[GoCadendIdx[results[:, 0].astype(np.int64)]], axis=1
                )
            else:
                dendLen = 0
            axonLen = np.linalg.norm(PFs[results[:, 3].astype(np.int64), 2:] - TJ[results[:, 1].astype(np.int32)], axis=1)
            del PFs
            if hasattr(h, "BREP_AAlenForPF"):
                axonLen = axonLen + h.BREP_AAlenForPF * np.linalg.norm(
                    GrC[results[:, 1].astype(np.int32)] - TJ[results[:, 1].astype(np.int32)], axis=1
                )
            else:
                axonLen = axonLen + np.linalg.norm(GrC[results[:, 1].astype(np.int32)] - TJ[results[:, 1].astype(np.int32)], axis=1)
            del GrC
            del TJ
            results = np.vstack(
                [
                    results[:, 1],
                    GoCadendIdx[results[:, 0].astype(np.int64)],
                    results[:, 2] + dendLen + axonLen,
                    adendSecIdx[results[:, 0].astype(np.int64)],
                    adendDendIdx[results[:, 0].astype(np.int64)],
                ]
            ).T
            del dendLen
            del axonLen
else:
    tree = KDTree(PFs[:, 2:])
    print(f"brep.py rank {MPIrank} KDtree built")
    K = min(len(PFs), K4T)
    distKDTree, idxKDTree = tree.query(GoCadend, k=K)
    print(f"brep.py rank {MPIrank} KDTree queried")
    del tree

    results = np.stack(
        [
            np.repeat(np.arange(len(GoCadend)), K).reshape(len(GoCadend), K),
            PFs[idxKDTree, 1],
            distKDTree,
            idxKDTree,
        ]
    ).transpose(1, 2, 0)
    del distKDTree
    del idxKDTree
    results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
    if results[:, 2].max() <= h.PFtoGoCzone:
        print(f"brep.py {MPIrank} all of {K} queries are within the zone")
    results = results[results[:, 2] <= h.PFtoGoCzone]
    results = results[np.unique(results[:, :2], axis=0, return_index=True)[1]]
    if hasattr(h, "BREP_dendcoeff") and h.BREP_dendcoeff > 0:
        dendLen = h.BREP_dendcoeff * np.linalg.norm(
            GoCadend[results[:, 0].astype(np.int32)] - GoC[GoCadendIdx[results[:, 0].astype(np.int64)]], axis=1
        )
    else:
        dendLen = 0
    axonLen = np.linalg.norm(PFs[results[:, 3].astype(np.int64), 2:] - TJ[results[:, 1].astype(np.int32)], axis=1)
    del PFs
    if hasattr(h, "BREP_AAlenForPF"):
        axonLen = axonLen + h.BREP_AAlenForPF * np.linalg.norm(GrC[results[:, 1].astype(np.int32)] - TJ[results[:, 1].astype(np.int32)], axis=1)
    else:
        axonLen = axonLen + np.linalg.norm(GrC[results[:, 1].astype(np.int32)] - TJ[results[:, 1].astype(np.int32)], axis=1)
    del GrC
    del TJ
    results = np.vstack(
        [
            results[:, 1],
            GoCadendIdx[results[:, 0].astype(np.int64)],
            results[:, 2] + dendLen + axonLen,
            adendSecIdx[results[:, 0].astype(np.int64)],
            adendDendIdx[results[:, 0].astype(np.int64)],
        ]
    ).T
    del dendLen
    del axonLen
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:, 1] % MPIsize == MPIrank]
vprint(f"brep.py rank {MPIrank} results computed")
vsave(f"PFtoGoCsources{MPIrank}.dat", results[:, 0], fmt="%d")
vsave(f"PFtoGoCtargets{MPIrank}.dat", results[:, 1], fmt="%d")
vsave(f"PFtoGoCdistances{MPIrank}.dat", results[:, 2], fmt="%f")
vsave(f"PFtoGoCsegments{MPIrank}.dat", results[:, [3, 4]], fmt="%d")
del results
vprint(f"brep.py rank {MPIrank} PFs to GoCs ({time.time() - tb:.2f} s)")


#######################################################
# GrC AAs to GoC AD/BD for glutamatergic synapses     #
#######################################################

tb = time.time()
if args.loadFiles:
    GrC = vload("GCcoordinates.sorted.dat")
    TJ = vload("GCTcoordinates.sorted.dat")
    GoC = vload("GoCcoordinates.sorted.dat")

    AAs = vload(f"AAcoordinates{MPIrank}.dat")
    AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T

    GoCbdend = vload("GoCbdendcoordinates.sorted.dat")
    bdendPerGoC = GoCbdend.shape[1] // 3
    GoCbdend = GoCbdend.reshape(GoCbdend.shape[0] * (GoCbdend.shape[1] // 3), 3)
    GoCbdendIdx = np.arange(len(GoCbdend)) // bdendPerGoC
    bdendSegIdx = np.arange(len(GoCbdend)) % bdendPerGoC
    bdendSecIdx = bdendSegIdx % (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts) // (h.GoC_Bd_nsegpts) + 1
    bdendDendIdx = h.numDendGolgi / 2 - bdendSegIdx // (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)
    del bdendSegIdx

    GoCdend = np.vstack([GoCadend, GoCbdend])
    GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
    dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
    dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])
else:
    if (
        os.path.isfile("GCcoordinates.sorted.dat")
        and os.path.isfile("GCTcoordinates.sorted.dat")
        and os.path.getsize("GCcoordinates.sorted.dat") > 0
        and os.path.getsize("GCTcoordinates.sorted.dat") > 0
    ):
        GrC = vload("GCcoordinates.sorted.dat")
        TJ = vload("GCTcoordinates.sorted.dat")
    else:
        GrC = vload("GCcoordinates.dat")
        TJ = vload("GCTcoordinates.dat")
    if os.path.isfile("GoCcoordinates.sorted.dat") and os.path.getsize("GoCcoordinates.sorted.dat") > 0:
        GoC = vload("GoCcoordinates.sorted.dat")
    else:
        GoC = vload("GoCcoordinates.dat")

    GrCidx = np.arange(MPIrank, len(GrC), MPIsize).astype(np.int64)

    # Ascending Axons (AAs)
    if hasattr(h, "BREP_AAlength") and h.BREP_AAlength > 0:
        # AA length is fixed in original BREP
        AAlength = h.BREP_AAlength * np.ones(len(GrCidx))
    else:
        AAlength = TJ[GrCidx, 2] - GrC[GrCidx, 2]
    del TJ
    numAAs = np.floor(AAlength / h.AAstep)
    AAs = np.stack(np.meshgrid(np.arange(numAAs.max()), AAlength / (numAAs - 1))).prod(axis=0)
    del numAAs
    AAs = np.vstack([np.arange(AAs.shape[0] * AAs.shape[1]) // AAs.shape[1], AAs.flatten()]).T
    AAs = AAs[AAs[:, 1] <= AAlength[AAs[:, 0].astype(np.int32)]]
    del AAlength
    GrCidx = GrCidx[AAs[:, 0].astype(np.int32)]
    AAs = np.vstack([GrCidx, GrC[GrCidx, :2].T, GrC[GrCidx, 2] + AAs[:, 1]]).T
    del GrCidx
    AAs = AAs[np.lexsort([AAs[:, 3], AAs[:, 2], AAs[:, 1]])].astype(np.float32)
    if args.saveAA:
        if args.testMode:
            vsave(f"pyAAcoordinates{MPIrank}.dat", AAs, fmt="%g")
        else:
            vsave(f"AAcoordinates{MPIrank}.dat", AAs, fmt="%g")
    AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T.astype(np.float32)

    # Basolateral dendrites (BDs)
    GoCbdend = []
    nseg = int(h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)
    target = np.vstack(
        [
            h.GoC_PhysBasolateralDendR * np.cos(theta[:, 0] * np.pi / 180),
            h.GoC_PhysBasolateralDendR * np.sin(theta[:, 0] * np.pi / 180),
            h.GoC_PhysBasolateralDendH * np.ones(len(theta)),
        ]
    ).T
    GoCbdend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
    target = np.vstack(
        [
            h.GoC_PhysBasolateralDendR * np.cos(theta[:, 1] * np.pi / 180),
            h.GoC_PhysBasolateralDendR * np.sin(theta[:, 1] * np.pi / 180),
            h.GoC_PhysBasolateralDendH * np.ones(len(theta)),
        ]
    ).T
    GoCbdend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
    del target
    GoCbdend = np.hstack(GoCbdend).reshape(len(GoC), nseg * 2 * 3)
    if args.saveDend:
        if MPIrank == 0:
            if args.testMode:
                vsave("pyGoCbdendcoordinates.sorted.dat", GoCbdend, fmt="%g")
            else:
                vsave("GoCbdendcoordinates.sorted.dat", GoCbdend, fmt="%g")
    bdendPerGoC = GoCbdend.shape[1] // 3
    GoCbdend = GoCbdend.reshape(GoCbdend.shape[0] * (GoCbdend.shape[1] // 3), 3)
    GoCbdendIdx = np.arange(len(GoCbdend)) // bdendPerGoC
    bdendSegIdx = np.arange(len(GoCbdend)) % bdendPerGoC
    bdendSecIdx = bdendSegIdx % (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts) // (h.GoC_Bd_nsegpts) + 1
    bdendDendIdx = h.numDendGolgi / 2 - bdendSegIdx // (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)
    del bdendSegIdx

    # dendrites
    GoCdend = np.vstack([GoCadend, GoCbdend])
    del GoCbdend
    GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
    del GoCbdendIdx
    dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
    del bdendSecIdx
    dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])
    del bdendDendIdx

vprint(f"brep.py rank {MPIrank} Preparation for AAs to GoCs ({time.time() - tb:.2f} s)")


vprint(f"brep.py rank {MPIrank} AAs to GoCs")
tb = time.time()
tree = KDTree(AAs[:, 2:])
print(f"brep.py rank {MPIrank} KDtree built")
K = min(len(AAs), K4T)
distKDTree, idxKDTree = tree.query(GoCdend, k=K)
print(f"brep.py rank {MPIrank} KDtree queried")
del tree

results = np.stack(
    [
        np.repeat(np.arange(len(GoCdend)), K).reshape(len(GoCdend), K),
        AAs[idxKDTree, 1],
        distKDTree,
        idxKDTree,
    ]
).transpose(1, 2, 0)
del distKDTree
del idxKDTree
results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
results = results[results[:, 2] <= h.AAtoGoCzone]
results = results[np.unique(results[:, :2], axis=0, return_index=True)[1]]
if hasattr(h, "BREP_dendcoeff") and h.BREP_dendcoeff > 0:
    dendLen = h.BREP_dendcoeff * np.linalg.norm(GoCdend[results[:, 0].astype(np.int64)] - GoC[GoCdendIdx[results[:, 0].astype(np.int64)]], axis=1)
else:
    dendLen = 0
axonLen = np.linalg.norm(AAs[results[:, 3].astype(np.int64), 2:] - GrC[results[:, 1].astype(np.int64)], axis=1)
del AAs
results = np.vstack(
    [
        results[:, 1],
        GoCdendIdx[results[:, 0].astype(np.int64)],
        results[:, 2] + dendLen + axonLen,
        dendSecIdx[results[:, 0].astype(np.int64)],
        dendDendIdx[results[:, 0].astype(np.int64)],
    ]
).T
del dendLen
del axonLen
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:, 1] % MPIsize == MPIrank]
vsave(f"AAtoGoCsources{MPIrank}.dat", results[:, 0], fmt="%d")
vsave(f"AAtoGoCtargets{MPIrank}.dat", results[:, 1], fmt="%d")
vsave(f"AAtoGoCdistances{MPIrank}.dat", results[:, 2], fmt="%f")
vsave(f"AAtoGoCsegments{MPIrank}.dat", results[:, [3, 4]], fmt="%d")
del results
vprint(f"brep.py rank {MPIrank} AAs to GoCs ({time.time() - tb:.2f} s)")


#######################################################
# GoC axon to GoC soma for GABAergic synapses         #
#######################################################

tb = time.time()
if args.loadFiles:
    GoC = vload("GoCcoordinates.sorted.dat")
    GoCaxon = vload("GoCaxoncoordinates.sorted.dat")
    axonPerGoC = GoCaxon.shape[1] // 3
    GoCaxon = GoCaxon.reshape(GoCaxon.shape[0] * (GoCaxon.shape[1] // 3), 3)
    GoCaxonIdx = np.arange(len(GoCaxon)) // axonPerGoC
    axonSegIdx = np.arange(len(GoCaxon)) % axonPerGoC
    axonSecIdx = axonSegIdx % (GoCAxonSegs * GoCAxonPts) // (GoCAxonPts) + 1
    axonDendIdx = h.numAxonGolgi - axonSegIdx // (GoCAxonSegs * GoCAxonPts)
    del axonSegIdx
else:
    if os.path.isfile("GoCcoordinates.sorted.dat") and os.path.getsize("GoCcoordinates.sorted.dat") > 0:
        GoC = vload("GoCcoordinates.sorted.dat")
    else:
        GoC = vload("GoCcoordinates.dat")
    np.random.seed(GoCaxonSeed)
    GoCaxon = np.random.random([len(GoC), int(h.numAxonGolgi), 3])
    GoCaxon[:, :, 0] = h.GoC_Axon_Xmin + np.floor((h.GoC_Axon_Xmax - h.GoC_Axon_Xmin + 1) * GoCaxon[:, :, 0])
    GoCaxon[:, :, 1] = h.GoC_Axon_Ymin + np.floor((h.GoC_Axon_Ymax - h.GoC_Axon_Ymin + 1) * GoCaxon[:, :, 1])
    GoCaxon[:, :, 2] = h.GoC_Axon_Zmin + np.floor((h.GoC_Axon_Zmax - h.GoC_Axon_Zmin + 1) * GoCaxon[:, :, 2])
    g = np.tile(GoC, int(h.numAxonGolgi)).reshape(len(GoC), int(h.numAxonGolgi), 3)
    GoCaxon = np.stack([g, g + GoCaxon], axis=2).reshape(len(GoC), int(h.numAxonGolgi * 2 * 3))
    del g
    if args.saveDend:
        if MPIrank == 0:
            if args.testMode:
                vsave("pyGoCaxoncoordinates.sorted.dat", GoCaxon, fmt="%g")
            else:
                vsave("GoCaxoncoordinates.sorted.dat", GoCaxon, fmt="%g")
    axonPerGoC = GoCaxon.shape[1] // 3
    GoCaxon = GoCaxon.reshape(GoCaxon.shape[0] * (GoCaxon.shape[1] // 3), 3)
    GoCaxonIdx = np.arange(len(GoCaxon)) // axonPerGoC
    axonSegIdx = np.arange(len(GoCaxon)) % axonPerGoC
    axonSecIdx = axonSegIdx % (GoCAxonSegs * GoCAxonPts) // (GoCAxonPts) + 1
    axonDendIdx = h.numAxonGolgi - axonSegIdx // (GoCAxonSegs * GoCAxonPts)
    del axonSegIdx


vprint(f"brep.py rank {MPIrank} Preparation for GoC axons to GoC somata ({time.time() - tb:.2f} s)")


vprint(f"brep.py rank {MPIrank} GoC axons to GoC somata")
tb = time.time()
idx = np.where(GoCaxonIdx % MPIsize == MPIrank)[0]
axonForTree = np.vstack([np.arange(len(idx)), GoCaxonIdx[idx], GoCaxon[idx].T]).T
del idx

tree = KDTree(axonForTree[:, 2:])
print(f"brep.py rank {MPIrank} KDtree built")
K = min(len(axonForTree), K4T)
distKDTree, idxKDTree = tree.query(GoC, k=K)
print(f"brep.py rank {MPIrank} KDtree queried")
del tree

results = np.stack(
    [
        np.repeat(np.arange(len(GoC)), K).reshape(len(GoC), K),
        axonForTree[idxKDTree, 1],
        distKDTree,
        idxKDTree,
    ]
).transpose(1, 2, 0)
del distKDTree
del idxKDTree
results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
results = results[
    (results[:, 2] <= h.GoCtoGoCzone)
    & (results[:, 0] != results[:, 1])
    & (GoC[results[:, 0].astype(np.int32), 2] < axonForTree[results[:, 3].astype(np.int64), 4])
]
results = results[np.unique(results[:, :2], axis=0, return_index=True)[1]]
dendLen = 0  # soma to soma
axonLen = np.linalg.norm(axonForTree[results[:, 3].astype(np.int64), 2:] - GoC[results[:, 1].astype(np.int32)], axis=1)
results = np.vstack([results[:, 1], results[:, 0], results[:, 2] + dendLen + axonLen]).T
del dendLen
del axonLen
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:, 1] % MPIsize == MPIrank]
vsave(f"GoCtoGoCsources{MPIrank}.dat", results[:, 0], fmt="%d")
vsave(f"GoCtoGoCtargets{MPIrank}.dat", results[:, 1], fmt="%d")
vsave(f"GoCtoGoCdistances{MPIrank}.dat", results[:, 2], fmt="%f")
vprint(f"brep.py rank {MPIrank} GoCs to GoCs inh ({time.time() - tb:.2f} s)")


#######################################################
# GoC soma to GoC soma for gap junction               #
#######################################################

vprint(f"brep.py rank {MPIrank} GoCs to GoCs gap")
tb = time.time()
GoCidx = np.arange(len(GoC))
idx = np.where(GoCidx % MPIsize == MPIrank)[0]
GoCForTree = np.vstack([np.arange(len(idx)), GoCidx[idx], GoC[idx].T]).T

tree = KDTree(GoCForTree[:, 2:])
print(f"brep.py rank {MPIrank} KDtree built")
K = min(len(GoCForTree), K4T)
distKDTree, idxKDTree = tree.query(GoC, k=K)
print(f"brep.py rank {MPIrank} KDtree queried")
del tree

results = np.stack(
    [
        np.repeat(np.arange(len(GoC)), K).reshape(len(GoC), K),
        GoCForTree[idxKDTree, 1],
        distKDTree,
        idxKDTree,
    ]
).transpose(1, 2, 0)
del distKDTree
del idxKDTree
results = results.reshape(results.shape[0] * results.shape[1], results.shape[2])
results = results[(results[:, 2] <= h.GoCtoGoCgapzone) & (results[:, 0] != results[:, 1])]
results = results[np.unique(results[:, :2], axis=0, return_index=True)[1]]
results = np.vstack([results[:, 1], results[:, 0], results[:, 2]]).T
results = np.vstack(MPIcomm.bcast(MPIcomm.gather(results, root=0), root=0))
results = results[results[:, 1] % MPIsize == MPIrank]
vsave(f"GoCtoGoCgapsources{MPIrank}.dat", results[:, 0], fmt="%d")
vsave(f"GoCtoGoCgaptargets{MPIrank}.dat", results[:, 1], fmt="%d")
vsave(f"GoCtoGoCgapdistances{MPIrank}.dat", results[:, 2], fmt="%f")
vprint(f"brep.py rank {MPIrank} GoCs to GoCs gap ({time.time() - tb:.2f} s)")


vprint(f"brep.py rank {MPIrank} completed ({time.time() - tb0:.2f} s)")


# #######################################################
# # Load/generate coordinates                           #
# #######################################################

# tb = time.time()
# if args.loadFiles:
#     vprint(f"brep.py rank {MPIrank} loadFiles: {args.loadFiles}")
#     GrC = vload("GCcoordinates.sorted.dat")
#     TJ = vload("GCTcoordinates.sorted.dat")
#     GoC = vload("GoCcoordinates.sorted.dat")

#     AAs = vload(f"AAcoordinates{MPIrank}.dat")
#     AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T

#     PFs = vload(f"PFcoordinates{MPIrank}.dat")
#     PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T

#     GoCadend = vload("GoCadendcoordinates.sorted.dat")
#     adendPerGoC = GoCadend.shape[1] // 3
#     GoCadend = GoCadend.reshape(GoCadend.shape[0] * (GoCadend.shape[1] // 3), 3)
#     GoCadendIdx = np.arange(len(GoCadend)) // adendPerGoC
#     adendSegIdx = np.arange(len(GoCadend)) % adendPerGoC
#     adendSecIdx = adendSegIdx % (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts) // (h.GoC_Ad_nsegpts) + 1
#     adendDendIdx = h.numDendGolgi - adendSegIdx // (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)

#     GoCbdend = vload("GoCbdendcoordinates.sorted.dat")
#     bdendPerGoC = GoCbdend.shape[1] // 3
#     GoCbdend = GoCbdend.reshape(GoCbdend.shape[0] * (GoCbdend.shape[1] // 3), 3)
#     GoCbdendIdx = np.arange(len(GoCbdend)) // bdendPerGoC
#     bdendSegIdx = np.arange(len(GoCbdend)) % bdendPerGoC
#     bdendSecIdx = bdendSegIdx % (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts) // (h.GoC_Bd_nsegpts) + 1
#     bdendDendIdx = h.numDendGolgi / 2 - bdendSegIdx // (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)

#     GoCdend = np.vstack([GoCadend, GoCbdend])
#     GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
#     dendSegIdx = np.hstack([adendSegIdx, bdendSegIdx])
#     dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
#     dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])

#     GoCaxon = vload("GoCaxoncoordinates.sorted.dat")
#     axonPerGoC = GoCaxon.shape[1] // 3
#     GoCaxon = GoCaxon.reshape(GoCaxon.shape[0] * (GoCaxon.shape[1] // 3), 3)
#     GoCaxonIdx = np.arange(len(GoCaxon)) // axonPerGoC
#     axonSegIdx = np.arange(len(GoCaxon)) % axonPerGoC
#     axonSecIdx = axonSegIdx % (GoCAxonSegs * GoCAxonPts) // (GoCAxonPts) + 1
#     axonDendIdx = h.numAxonGolgi - axonSegIdx // (GoCAxonSegs * GoCAxonPts)
#     vprint(f"brep.py rank {MPIrank} Finish to load files ({time.time() - tb:.2f} s)")

# else:
#     vprint(f"brep.py rank {MPIrank} loadFiles: {args.loadFiles}")
#     if (
#         os.path.isfile("GCcoordinates.sorted.dat")
#         and os.path.isfile("GCTcoordinates.sorted.dat")
#         and os.path.getsize("GCcoordinates.sorted.dat") > 0
#         and os.path.getsize("GCTcoordinates.sorted.dat") > 0
#     ):
#         GrC = vload("GCcoordinates.sorted.dat")
#         TJ = vload("GCTcoordinates.sorted.dat")
#     else:
#         GrC = vload("GCcoordinates.dat")
#         TJ = vload("GCTcoordinates.dat")
#     if os.path.isfile("GoCcoordinates.sorted.dat") and os.path.getsize("GoCcoordinates.sorted.dat") > 0:
#         GoC = vload("GoCcoordinates.sorted.dat")
#     else:
#         GoC = vload("GoCcoordinates.dat")
#     MPIcomm.barrier()
#     if MPIrank == 0:
#         vsave("GCcoordinates.sorted.dat", GrC, "%g")
#         vsave("GCTcoordinates.sorted.dat", TJ, "%g")
#         vsave("GoCcoordinates.sorted.dat", GoC, "%g")

#     GrCidx = np.arange(MPIrank, len(GrC), MPIsize).astype(np.int64)

#     # Ascending Axons (AAs)
#     if hasattr(h, "BREP_AAlength") and h.BREP_AAlength > 0:
#         # AA length is fixed in original BREP
#         AAlength = h.BREP_AAlength * np.ones(len(GrCidx))
#     else:
#         AAlength = TJ[GrCidx, 2] - GrC[GrCidx, 2]
#     numAAs = np.floor(AAlength / h.AAstep)
#     AAs = np.stack(np.meshgrid(np.arange(numAAs.max()), AAlength / (numAAs - 1))).prod(axis=0)
#     del numAAs
#     AAs = np.vstack([np.arange(AAs.shape[0] * AAs.shape[1]) // AAs.shape[1], AAs.flatten()]).T
#     AAs = AAs[AAs[:, 1] <= AAlength[AAs[:, 0].astype(np.int32)]]
#     i = GrCidx[AAs[:, 0].astype(np.int32)]
#     AAs = np.vstack([i, GrC[i, :2].T, GrC[i, 2] + AAs[:, 1]]).T
#     del i
#     AAs = AAs[np.lexsort([AAs[:, 3], AAs[:, 2], AAs[:, 1]])].astype(np.float32)
#     if args.saveAA:
#         if args.testMode:
#             vsave(f"pyAAcoordinates{MPIrank}.dat", AAs, fmt="%g")
#         else:
#             vsave(f"AAcoordinates{MPIrank}.dat", AAs, fmt="%g")
#     AAs = np.vstack([np.arange(len(AAs)), AAs.T]).T.astype(np.float32)

#     # Parallel Fibres (PFs)
#     if hasattr(h, "BREP_PFcoeff") and h.BREP_PFcoeff > 0:
#         numPFs = np.floor(2 * h.PFlength * h.BREP_PFcoeff / h.PFstep).astype(np.int32)
#         PFs = np.linspace(-h.PFlength * h.BREP_PFcoeff, h.PFlength * h.BREP_PFcoeff, numPFs).astype(np.float32)
#     else:
#         numPFs = np.floor(2 * h.PFlength / h.PFstep).astype(np.int32)
#         PFs = np.linspace(-h.PFlength, h.PFlength, numPFs).astype(np.float32)
#     del numPFs
#     PFs = np.tile(PFs, [len(GrCidx), 1]).astype(np.float32)
#     PFs = np.vstack([GrCidx[np.arange(PFs.shape[0] * PFs.shape[1]) // PFs.shape[1]], PFs.flatten()]).T.astype(np.float32)
#     PFs = np.vstack([PFs[:, 0], TJ[PFs[:, 0].astype(np.int32), 0] + PFs[:, 1], TJ[PFs[:, 0].astype(np.int32), 1:].T]).T.astype(np.float32)
#     PFs = PFs[np.lexsort([PFs[:, 3], PFs[:, 2], PFs[:, 1]])].astype(np.float32)
#     if args.savePF:
#         if args.testMode:
#             vsave(f"pyPFcoordinates{MPIrank}.dat", PFs, fmt="%g")
#         else:
#             vsave(f"PFcoordinates{MPIrank}.dat", PFs, fmt="%g")
#     PFs = np.vstack([np.arange(len(PFs)), PFs.T]).T.astype(np.float32)

#     # random numbers for GoC dendrites
#     if args.randomTable:
#         from itertools import cycle

#         theta = cycle(vload(args.randomTable))
#         theta = [next(theta) for i in range(len(GoC) * int(h.numDendGolgi))]
#         theta = np.array(theta).reshape([len(GoC), int(h.numDendGolgi)])
#     else:
#         np.random.seed(GoCdendSeed)
#         theta = np.random.normal(0, 1, [len(GoC), int(h.numDendGolgi)])
#     thetaStd = np.array([h.GoC_Btheta_stdev, h.GoC_Btheta_stdev, h.GoC_Atheta_stdev, h.GoC_Atheta_stdev])
#     thetaMean = np.array([h.GoC_Btheta_max, h.GoC_Btheta_min, h.GoC_Atheta_max, h.GoC_Atheta_min])
#     theta = theta * thetaStd + thetaMean
#     del thetaMean
#     del thetaStd

#     # Basolateral dendrites (BDs)
#     GoCbdend = []
#     nseg = int(h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)
#     target = np.vstack(
#         [
#             h.GoC_PhysBasolateralDendR * np.cos(theta[:, 0] * np.pi / 180),
#             h.GoC_PhysBasolateralDendR * np.sin(theta[:, 0] * np.pi / 180),
#             h.GoC_PhysBasolateralDendH * np.ones(len(theta)),
#         ]
#     ).T
#     GoCbdend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
#     target = np.vstack(
#         [
#             h.GoC_PhysBasolateralDendR * np.cos(theta[:, 1] * np.pi / 180),
#             h.GoC_PhysBasolateralDendR * np.sin(theta[:, 1] * np.pi / 180),
#             h.GoC_PhysBasolateralDendH * np.ones(len(theta)),
#         ]
#     ).T
#     GoCbdend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
#     del target
#     GoCbdend = np.hstack(GoCbdend).reshape(len(GoC), nseg * 2 * 3)
#     if args.saveDend:
#         if MPIrank == 0:
#             if args.testMode:
#                 vsave("pyGoCbdendcoordinates.sorted.dat", GoCbdend, fmt="%g")
#             else:
#                 vsave("GoCbdendcoordinates.sorted.dat", GoCbdend, fmt="%g")
#     bdendPerGoC = GoCbdend.shape[1] // 3
#     GoCbdend = GoCbdend.reshape(GoCbdend.shape[0] * (GoCbdend.shape[1] // 3), 3)
#     GoCbdendIdx = np.arange(len(GoCbdend)) // bdendPerGoC
#     bdendSegIdx = np.arange(len(GoCbdend)) % bdendPerGoC
#     bdendSecIdx = bdendSegIdx % (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts) // (h.GoC_Bd_nsegpts) + 1
#     bdendDendIdx = h.numDendGolgi / 2 - bdendSegIdx // (h.GoC_Bd_nseg * h.GoC_Bd_nsegpts)

#     # Apical dendrites (ADs)
#     GoCadend = []
#     nseg = int(h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)
#     target = np.vstack(
#         [
#             h.GoC_PhysApicalDendR * np.cos(theta[:, 2] * np.pi / 180),
#             h.GoC_PhysApicalDendR * np.sin(theta[:, 2] * np.pi / 180),
#             h.GoC_PhysApicalDendH * np.ones(len(theta)),
#         ]
#     ).T
#     GoCadend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
#     target = np.vstack(
#         [
#             h.GoC_PhysApicalDendR * np.cos(theta[:, 3] * np.pi / 180),
#             h.GoC_PhysApicalDendR * np.sin(theta[:, 3] * np.pi / 180),
#             h.GoC_PhysApicalDendH * np.ones(len(theta)),
#         ]
#     ).T
#     del theta
#     GoCadend.append(np.linspace(GoC, GoC + target, nseg).transpose(1, 0, 2))
#     del target
#     GoCadend = np.hstack(GoCadend).reshape(len(GoC), nseg * 2 * 3)
#     if args.saveDend:
#         if MPIrank == 0:
#             if args.testMode:
#                 vsave("pyGoCadendcoordinates.sorted.dat", GoCadend, fmt="%g")
#             else:
#                 vsave("GoCadendcoordinates.sorted.dat", GoCadend, fmt="%g")
#     adendPerGoC = GoCadend.shape[1] // 3
#     GoCadend = GoCadend.reshape(GoCadend.shape[0] * (GoCadend.shape[1] // 3), 3)
#     GoCadendIdx = np.arange(len(GoCadend)) // adendPerGoC
#     adendSegIdx = np.arange(len(GoCadend)) % adendPerGoC
#     adendSecIdx = adendSegIdx % (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts) // (h.GoC_Ad_nsegpts) + 1
#     adendDendIdx = h.numDendGolgi - adendSegIdx // (h.GoC_Ad_nseg * h.GoC_Ad_nsegpts)

#     # dendrites
#     GoCdend = np.vstack([GoCadend, GoCbdend])
#     del GoCbdend
#     GoCdendIdx = np.hstack([GoCadendIdx, GoCbdendIdx])
#     del GoCbdendIdx
#     dendSegIdx = np.hstack([adendSegIdx, bdendSegIdx])
#     del adendSegIdx
#     del bdendSegIdx
#     dendSecIdx = np.hstack([adendSecIdx, bdendSecIdx])
#     del bdendSecIdx
#     dendDendIdx = np.hstack([adendDendIdx, bdendDendIdx])
#     del bdendDendIdx

#     # GoC axons
#     np.random.seed(GoCaxonSeed)
#     GoCaxon = np.random.random([len(GoC), int(h.numAxonGolgi), 3])
#     GoCaxon[:, :, 0] = h.GoC_Axon_Xmin + np.floor((h.GoC_Axon_Xmax - h.GoC_Axon_Xmin + 1) * GoCaxon[:, :, 0])
#     GoCaxon[:, :, 1] = h.GoC_Axon_Ymin + np.floor((h.GoC_Axon_Ymax - h.GoC_Axon_Ymin + 1) * GoCaxon[:, :, 1])
#     GoCaxon[:, :, 2] = h.GoC_Axon_Zmin + np.floor((h.GoC_Axon_Zmax - h.GoC_Axon_Zmin + 1) * GoCaxon[:, :, 2])
#     g = np.tile(GoC, int(h.numAxonGolgi)).reshape(len(GoC), int(h.numAxonGolgi), 3)
#     GoCaxon = np.stack([g, g + GoCaxon], axis=2).reshape(len(GoC), int(h.numAxonGolgi * 2 * 3))
#     del g
#     if args.saveDend:
#         if MPIrank == 0:
#             if args.testMode:
#                 vsave("pyGoCaxoncoordinates.sorted.dat", GoCaxon, fmt="%g")
#             else:
#                 vsave("GoCaxoncoordinates.sorted.dat", GoCaxon, fmt="%g")
#     axonPerGoC = GoCaxon.shape[1] // 3
#     GoCaxon = GoCaxon.reshape(GoCaxon.shape[0] * (GoCaxon.shape[1] // 3), 3)
#     GoCaxonIdx = np.arange(len(GoCaxon)) // axonPerGoC
#     axonSegIdx = np.arange(len(GoCaxon)) % axonPerGoC
#     axonSecIdx = axonSegIdx % (GoCAxonSegs * GoCAxonPts) // (GoCAxonPts) + 1
#     axonDendIdx = h.numAxonGolgi - axonSegIdx // (GoCAxonSegs * GoCAxonPts)

#     vprint(f"brep.py rank {MPIrank} Finish to create coordinations ({time.time() - tb:.2f} s)")

# if args.testMode:
#     print("TEST MODE")
#     sys.exit(1)
