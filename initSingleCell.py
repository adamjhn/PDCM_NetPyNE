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
    simConfigDefault="cfgSS.py", netParamsDefault="netParamsSingleCell.py"
)
sim.initialize(
    simConfig=simConfig, netParams=netParams
)  # create network object and set cfg and net params
sim.net.createPops()  # instantiate network populations
sim.net.createCells()  # instantiate network cells based on defined populations
sim.net.addStims()  # add network stimulation
# fih = h.FInitializeHandler(2, lambda: fi(sim.net.cells))
sim.net.addRxD(nthreads=1)
sim.net.connectCells()  # create connections between cells based on params
sim.setupRecording()  # setup variables to record for each cell (spikes, V traces, etc)
sim.runSim()  # run parallel Neuron simulation
sim.gatherData()  # gather spiking data and cell info from each node
sim.saveData()
# 	#sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [100,600], 'maxFreq':80, 'figSize': (8,4), 'fontSize': 16, 'saveData': False, 'saveFig': cfg.saveFolder+cfg.simLabel+'_LFP_spec_elec_'+str(elec)+'.png', 'showFig': False})
