'''
NetPyNE version of Potjans and Diesmann thalamocortical network

cfg.py -- contains the simulation configuration (cfg object)

Modified to include RxD for K, Na, Cl and O2.

'''

from netpyne import specs
import numpy as np

############################################################
#
#                    SIMULATION CONFIGURATION
#
############################################################

cfg = specs.SimConfig() # object of class SimConfig to store simulation configuration

############################################################
# Run options
############################################################

cfg.seeds['stim']=3
cfg.duration = 1e3   # Duration of the simulation, in ms
cfg.dt = 0.025          # Internal integration timestep to use
cfg.verbose = 0     # Show detailed messages
cfg.seeds['m'] = 123
cfg.printPopAvgRates = True
cfg.printRunTime = 1
cfg.hParams['celsius'] = 34

### Options to save memory in large-scale ismulations
cfg.gatherOnlySimData = False  #Original

# set the following 3 options to False when running large-scale versions of the model (>50% scale) to save memory
cfg.saveCellSecs = True 
cfg.saveCellConns = True
cfg.createPyStruct = True

 # Network dimensions
cfg.sizeX = 242.0 #250.0 #1000
cfg.sizeY = 1470.0 #250.0 #1000
cfg.sizeZ = 242.0 #200.0
cfg.density = 90000.0
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ
cfg.borderX = [0, 72.960082572]
cfg.borderY = [-119.72972477280001, 189.30002448484998]
cfg.borderZ = [0, 0]

# slice conditions 
cfg.ox = 'perfused'
if cfg.ox == 'perfused':
    cfg.o2_bath = 0.1
    cfg.alpha_ecs = 0.2 
    cfg.tort_ecs = 1.6
elif cfg.ox == 'hypoxic':
    cfg.o2_bath = 0.01
    cfg.alpha_ecs = 0.07 
    cfg.tort_ecs = 1.8

cfg.sa2v = 3.0 # False

cfg.betaNrn = 0.24
cfg.Ncell = int(cfg.density*(cfg.sizeX*cfg.sizeY*cfg.sizeZ*1e-9)) # default 90k / mm^3
if cfg.density == 90000.0:
    cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.Ncell)) ** (1/3)
else:
    cfg.rs = 7.52

cfg.epas = -70 # False
cfg.gpas = 0.0001
cfg.sa2v = 3.0 # False
if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1/2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params 
cfg.k0 = 3.5
cfg.r0 = 100.0


# BPO config
#cfg.update_params = True
#cfg.secmap = {'somatic':['soma'], 'apical':['Adend1','Adend2','Adend3'], 'axonal':['axon'], 'basal':['Bdend']}

# Scale synapses weights
cfg.excWeight = 1.0
cfg.inhWeight = 1.0

###########################################################
# Network Options
###########################################################

# DC=True ;  TH=False; Balanced=True   => Reproduce Figure 7 A1 and A2
# DC=False;  TH=False; Balanced=False  => Reproduce Figure 7 B1 and B2
# DC=False ; TH=False; Balanced=True   => Reproduce Figure 8 A, B, C and D
# DC=False ; TH=False; Balanced=True   and run to 60 s to => Table 6 
# DC=False ; TH=True;  Balanced=True   => Figure 10A. But I want a partial reproduce so I guess Figure 10C is not necessary

# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 0.01  # 1.0 = 80.000 

# External input DC or Poisson
cfg.DC = False #True = DC // False = Poisson

# Thalamic input in 4th and 6th layer on or off
cfg.TH = False #True = on // False = off

# Balanced and Unbalanced external input as PD article
cfg.Balanced = True #True=Balanced // False=Unbalanced

# Scaling factor for weights when replacing point neurons with multicompartment neurons
cfg.scaleConnWeight = 0.000001

cfg.simLabel = 'pd_mc_scale-%s_DC-%d_TH-%d_Balanced-%d_dur-%d_wscale_%.6g'%(str(cfg.ScaleFactor), int(cfg.DC), int(cfg.TH), int(cfg.Balanced), int(cfg.duration/1e3), cfg.scaleConnWeight)

###########################################################
# Recording and plotting options
###########################################################

cfg.recordStep = 0.1         # Step size in ms to save data (e.g. V traces, LFP, etc)
cfg.filename = cfg.simLabel  # Set file output name
cfg.saveFolder = 'dataSS/'
cfg.savePickle = True         # Save params, network and sim output to pickle file
cfg.saveJson = False
cfg.recordStim = False
cfg.printSynsAfterRule = False
cfg.recordCells = [f'L{i}{ei}_{idx}' for i in [2,4,5,6] for ei in ['e','i'] for idx in range(10)]
cfg.recordTraces = {f'{var}_soma':{'sec':'soma','loc':0.5,'var':var} for var in ['v','nai', 'ki', 'cli', 'dumpi']}  # Dict with traces to record
#cfg.analysis['plotRaster'] = {'saveFig': True}                  # Plot a raster
#cfg.analysis['plotTraces'] = {'saveFig': True}  # Plot recorded traces for this list of cells
cfg.recordCellsSpikes = [f"{pop}_{idx}" for pop in ['L2e', 'L2i', 'L4e', 'L4i', 'L5e', 'L5i','L6e', 'L6i'] for idx in range(10)] # record only spikes of cells (not ext stims)


# # raster plot 
# cfg.analysis['plotRaster'] = {'include': cfg.recordCellsSpikes, 'timeRange': [100,600], 'popRates' : False, 'figSize' : (6,12),  
# 	'labels':'overlay', 'orderInverse': True, 'fontSize': 16, 'dpi': 300, 'showFig': False, 'saveFig': True}

# # statistics plot (include update in netParams.py)
# cfg.analysis['plotSpikeStats'] = {'include': cfg.recordCellsSpikes, 'stats' : ['rate'], 'xlim': [0,15], 'legendLabels': cfg.recordCellsSpikes,
# 	'timeRange' : [100,600], 'fontSize': 20, 'dpi': 300, 'figSize': (3,12),'showFig':False, 'saveFig': True}

# # # plot traces
# cfg.recordTraces = {'V_soma': {'sec':'soma','loc':0.5, 'var':'v'}}

# cfg.analysis['plotTraces'] = {'include': [('L2e', 0),('L2i', 0), ('L4e', 0),('L4i', 0), ('L5e', 0), ('L5i', 0), ('L6e', 0), ('L6i', 0)], 
# 							'timeRange': [100,600], 'figSize': (6,3), 'legend': False, 'fontSize': 16, 'overlay': True, 'axis': False, 'oneFigPer': 'trace', 'showFig': False, 'saveFig': True}

# cfg.analysis['plotLFP'] = {'plots': ['timeSeries'], 'electrodes': range(15), 'timeRange': [100,600], 'fontSize': 20, 'maxFreq':80, 'figSize': (6,12), 'dpi': 300, 'saveData': False, 'saveFig': True, 'showFig': False}

layer_bounds = {'L1': 0.08*1470, 'L2': 0.27*1470, 'L4': 0.58*1470, 'L5': 0.73*1470, 'L6': 1.0*1470}



# cfg.analysis['plotShape'] = {'includePost': cfg.recordCellsSpikes, 'includeAxon': 1, 'cvar': 'voltage', 'fontSize': 16, 'figSize': (12,8), 
#							'axis': 'on', 'axisLabels': False, 'saveFig': True, 'dpi': 300, 'dist': 0.65}

# plot 2D net structure
# cfg.analysis['plot2Dnet'] = {'include': cfg.recordCellsSpikes, 'saveFig': True,  'figSize': (10,15)}

# plot convergence connectivity as 2D 
# cfg.analysis['plotConn'] = {'includePre': cfg.recordCellsSpikes, 'includePost': cfg.recordCellsSpikes, 'feature': 'convergence', \
#    'synOrConn': 'conn', 'graphType': 'bar', 'saveFig': True, 'figSize': (15, 9)}

# plot firing rate spectrogram  (run for 4 sec)
# cfg.analysis['plotRateSpectrogram'] = {'include': ['allCells'], 'saveFig': True, 'figSize': (15, 7)}

# plot granger causality (run for 4 sec)
# cfg.analysis.granger = {'cells1': ['L2i'], 'cells2': ['L4e'], 'label1': 'L2i', 'label2': 'L4e', 'timeRange': [500,4000], 'saveFig': True, 'binSize': 4}

