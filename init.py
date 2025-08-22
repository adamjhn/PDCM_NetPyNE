'''
NetPyNE version of Potjans and Diesmann thalamocortical network

init.py -- code to run the simulation

to compile mod files: 
	nrnivmodl  
to run on single core: 
	python init.py
to run on multiple cores: 
	mpiexec -n 2 nrniv -python -mpi init.py 
'''

import matplotlib; matplotlib.use('Agg')  # to avoid graphics error in servers
import numpy as np
import json
from netpyne import sim
from neuron import h
from cfg import cfg
from netParams import netParams
import pickle
import os
from stats import networkStatsFromSim
import sys

############################################################
#               Create network and run simulation
############################################################


seed = float(sys.argv[-1])

sim.initialize(
    simConfig = cfg,  
    netParams = netParams)          # create network object and set cfg and net params
cfg.seeds['stim'] += seed
cfg.seeds['m'] += seed
cfg.saveFolder = f"{cfg.saveFolder}_{cfg.seeds['stim']}_{cfg.seeds['m']}"
print(cfg.saveFolder)


pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()
pc.timeout(0)
pc.set_maxstep(100) 

sim.net.createPops()                    # instantiate network populations
sim.net.createCells()                   # instantiate network cells based on defined populations

# randomize m parameter of cells
rand=h.Random()
for c in sim.net.cells:
	if c.tags['cellModel'] == 'IntFire_PD':
		rand.Random123(c.gid, cfg.seeds['m'])
		c.hPointp.m = rand.normal(-58,10)

sim.net.addStims()              # add network stimulation
sim.net.connectCells()                  # create connections between cells based on params
sim.setupRecording()                    # setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()                            # run parallel Neuron simulation  
sim.gatherData()                        # gather spiking data and cell info from each node
sim.saveData()                          # save params, cell info and sim output to file (pickle,mat,txt,etc)#
sim.analysis.plotData()               # plot spike raster etc

if pcid == 0:
    networkStatsFromSim(sim, filename=os.path.join(cfg.saveFolder, "netstats.json"))

def getInputVector(gid, fixedDelay=0):
    inputs = {c['preGid']:c for c in sim.net.cells[gid].conns}
    spikes = np.array([t + inputs[idx]['delay'] - fixedDelay for t,idx in zip(spkt,spkid) if idx in inputs])
    weights = np.array([inputs[idx]['weight'] for idx in spkid if idx in inputs])
    mechs = np.array([inputs[idx]['synMech'] for idx in spkid if idx in inputs])
    secs = np.array([inputs[idx]['sec'] for idx in spkid if idx in inputs])
    locs = np.array([inputs[idx]['loc'] for idx in spkid if idx in inputs])

    I = np.argsort(spikes)
   
   
    return spikes[I], weights[I], mechs[I], secs[I], locs[I]
spkt, spkid = sim.simData['spkt'].as_numpy(), sim.simData['spkid'].as_numpy()
pops = ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i']
sample = {}
N = 5 
fixedDelay = 0 # delay NetStim that replaces original inputs will have
for p in pops:
    sample[p] = {'cellGids': [], 'inputs':[], 'weight':[], 'mechs':[], 'secs':[], 'locs':[], 'output':[], 'rates':[]}
    while len(sample[p]['cellGids']) < N:
        idx = np.random.randint(0,len(getattr(sim.net.pops,p).cellGids))
        gid = getattr(sim.net.pops,p).cellGids[idx]
        if gid not in sample[p]['cellGids']:
            s, w, mlst, slst, llst = getInputVector(gid, fixedDelay=fixedDelay)
            #print(p, gid, len(sample[p]['cellGids']), len(s))
            if len(s) > 0:
                sample[p]['cellGids'].append(gid)
                sample[p]['inputs'].append(s)
                sample[p]['weight'].append(w)
                sample[p]['mechs'].append(mlst)
                sample[p]['secs'].append(slst)
                sample[p]['locs'].append(llst)
                resp = spkt[spkid==gid]
                sample[p]['output'].append(resp)
                sample[p]['rates'].append(1e3*len(resp)/sim.cfg.duration)

pickle.dump(sample, open(os.path.join(sim.cfg.saveFolder,f"sample_{sim.cfg.simLabel}.pkl"),'wb'))
