"""
NetPyNE version of Potjans and Diesmann thalamocortical network

init.py -- code to run the simulation
"""

from netpyne import sim
from neuron import h
import pandas as pd
import numpy as np


############################################################
#               Create network and run simulation
############################################################


simConfig, netParams = sim.readCmdLineArgs(
    simConfigDefault="cfgSS.py", netParamsDefault="netParamsSS.py"
)
sim.initialize(
    simConfig=simConfig, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations
sim.net.addStims()  # add network stimulation
sim.net.addRxD(nthreads=6)
"""
df = pd.read_json('PDMCExample.json')

L = list(df.columns)
N_Full = np.array([len(df[pop]['cellGids']) for pop in L])
counts = {pop:0 for pop in L}
for gid in range(N_Full).sum()):
    cell = sim.cellByGid(gid)
    pop = cell.tags['cellType']
    idx = counts[pop]
    counts[pop] += 1
    inp = df[pop]['inputs'][idx]
    typ = df[pop]['mech'][idx]
    excVec = h.Vector([t for t,m in zip(inp,typ) if typ == 'exc'])
    inhVec = h.Vector([t for t,m in zip(inp,typ) if typ == 'inh'])
"""
sim.net.connectCells()  # create connections between cells based on params
sim.setupRecording()  # setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()  # run parallel Neuron simulation
sim.gatherData()  # gather spiking data and cell info from each node
sim.saveData()  # save params, cell info and sim output to file (pickle,mat,txt,etc)#
# sim.analysis.plotData()               # plot spike raster etc


# # Plot all electrodes separately; use electrode 6
# for elec in [3]: #range(15):
# 	sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [100,600], 'maxFreq':80, 'figSize': (7,4), 'fontSize': 16, 'saveData': False, 'saveFig': cfg.saveFolder+cfg.simLabel+'_LFP_PSD_elec_'+str(elec)+'.png', 'showFig': False})
# 	#sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [100,600], 'maxFreq':80, 'figSize': (8,4), 'fontSize': 16, 'saveData': False, 'saveFig': cfg.saveFolder+cfg.simLabel+'_LFP_spec_elec_'+str(elec)+'.png', 'showFig': False})
