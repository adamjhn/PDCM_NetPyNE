"""
NetPyNE version of Potjans and Diesmann thalamocortical network

cfg.py -- contains the simulation configuration (cfg object)

Modified to include RxD for K, Na, Cl and O2.

"""

from netpyne import specs
import numpy as np

############################################################
#
#                    SIMULATION CONFIGURATION
#
############################################################

cfg = specs.SimConfig()  # object of class SimConfig to store simulation configuration

############################################################
# Run options
############################################################

cfg.seeds["stim"] = 3
cfg.duration = 1000  # Duration of the simulation, in ms
cfg.dt = 0.025  # Internal integration timestep to use
cfg.verbose = False  # Show detailed messages
cfg.seeds["m"] = 123
cfg.popOpt = ["L2e"]
cfg.printPopAvgRates = False
cfg.hParams["celsius"] = 37
cfg.hParams["v_init"] = -70
cfg.cvode_active = False
# scaling factors
cfg.poissonRateFactor = 1.0
cfg.connected = True
### Options to save memory in large-scale ismulations
cfg.gatherOnlySimData = True  # Original
cfg.random123 = True


# Size of Network. Adjust this constants, please!
cfg.ScaleFactor = 0.16  # 1.0 = 80.000
cfg.scaleConnWeightNetStims = 1
cfg.scaleConnWeightNetStimStd = 1

# set the following 3 options to False when running large-scale versions of the model (>50% scale) to save memory
cfg.saveCellSecs = True
cfg.saveCellConns = True
cfg.createPyStruct = True
cfg.printPopAvgRates = True
cfg.singleCells = False  # create one cell in each population
cfg.printRunTime = False
cfg.Kceil = 15.0
cfg.nRec = 25
cfg.cellPops = [
    "L2e",
    "L2i",
    "L4e",
    "L4i",
    "L5e",
    "L5i",
    "L6e",
    "L6i",
]  # record only spikes of cells (not ext stims)
cfg.cellPopsInit = (-85, -60)
cfg.recordCellsSpikes = [f"{pop}_{idx}" for pop in cfg.popOpt for idx in range(10)]

if cfg.recordStim:
    cfg.recordCellsSpikes += [
        f"poissL{i}{ei}" for i in [2, 4, 5, 6] for ei in ["e", "i"]
    ]
    cfg.recordCellsSpikes += [f"bkg_THL{i}{ei}" for i in [4, 6] for ei in ["e", "i"]]

# cfg.recordCells = [
#    (f"L{i}{ei}", idx) for i in [2, 4, 5, 6] for ei in ["e", "i"] for idx in range(10)
# ]

cfg.recordCells = [f"{pop}_{idx}" for pop in cfg.popOpt for idx in range(10)]
cfg.recordTraces = {
    f"{var}_soma": {"sec": "soma", "loc": 0.5, "var": var}
    for var in ["v", "nai", "ki", "cli", "o2_consumedo"]
}
"""
cfg.recordTraces["exc_i"] = {"sec":'soma', "loc":0.5, "synMech":"exc", "var":"i", 'index':0}
cfg.recordTraces["inh_i"] = {"sec":'soma', "loc":0.5, "synMech":"inh", "var":"i", 'index':0}
cfg.recordTraces["exc_g"] = {"sec":'soma', "loc":0.5, "synMech":"exc", "var":"g", 'index':0}
cfg.recordTraces["inh_g"] = {"sec":'soma', "loc":0.5, "synMech":"inh", "var":"g", 'index':0}
"""

cfg.seed = 0
cfg.seeds = {
    "conn": 2 + cfg.seed,
    "stim": 3 + cfg.seed,
    "loc": 4 + cfg.seed,
    "cell": 5 + cfg.seed,
    "rec": 1 + cfg.seed,
}
# Network dimensions
cfg.sizeX = 700  # 250.0 #1000
cfg.sizeY = 2131.2851  # 1470.0  # 250.0 #1000
cfg.sizeZ = 700  # 200.0
cfg.dx = 700
cfg.Vtissue = cfg.sizeX * cfg.sizeY * cfg.sizeZ

# slice conditions
cfg.o2_bath = 0.06
cfg.o2_init = 0.04
cfg.alpha_ecs = 0.2
cfg.alpha_ecs = 0.2
cfg.tort_ecs = 1.6
cfg.o2drive = 0.013
cfg.ox = "perfused"

cfg.sa2v = 3.4  # False

cfg.betaNrn = 0.29
cfg.N_Full = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]
cfg.Ncell = sum([max(1, int(i * cfg.ScaleFactor)) for i in cfg.N_Full])
cfg.rs = ((cfg.betaNrn * cfg.Vtissue) / (2 * np.pi * cfg.Ncell)) ** (1 / 3)

cfg.epas = -70.00000000000013
cfg.Cm = 1.0
cfg.Ra = 100
if cfg.sa2v:
    cfg.somaR = (cfg.sa2v * cfg.rs**3 / 2.0) ** (1 / 2)
else:
    cfg.somaR = cfg.rs
cfg.cyt_fraction = cfg.rs**3 / cfg.somaR**3

# sd init params
cfg.k0 = 3.5
cfg.r0 = 100.0


# Scale synapses weights -- optimized
cfg.excWeight = 0.01  # 0.9170195634091205
cfg.inhWeightScale = 10  # 9.826449573438962


cfg.weightMin = 0.1
cfg.dWeight = 0.1
# optimized single cell parameters

# single cell optimized with AP peak >= 30mV
cfg.gnabar = 0.02211617598652266
cfg.gkbar = 0.004001629507118593
cfg.ukcc2 = 0.0019830617654271222
cfg.unkcc1 = 6.506198176269446
cfg.pmax = 3  # 5035.941975532757
cfg.gpas = 4.2407540290597475e-05

cfg.gkleak_scale = 1
cfg.KKo = 5.3
cfg.KNai = 27.9
cfg.GliaKKo = 3.5  # 4.938189537703508  # originally 3.5 mM
cfg.GliaPumpScale = 1 / 3  # 1 / 3  # originally 1/3
cfg.scaleConnWeight = 1


###########################################################
# Network Options
###########################################################

# DC=True ;  TH=False; Balanced=True   => Reproduce Figure 7 A1 and A2
# DC=False;  TH=False; Balanced=False  => Reproduce Figure 7 B1 and B2
# DC=False ; TH=False; Balanced=True   => Reproduce Figure 8 A, B, C and D
# DC=False ; TH=False; Balanced=True   and run to 60 s to => Table 6
# DC=False ; TH=True;  Balanced=True   => Figure 10A. But I want a partial reproduce so I guess Figure 10C is not necessary


# External input DC or Poisson
cfg.DC = False  # True = DC // False = Poisson

# Thalamic input in 4th and 6th layer on or off
cfg.TH = True  # True = on // False = off

# Balanced and Unbalanced external input as PD article
cfg.Balanced = True  # True=Balanced // False=Unbalanced

"""
# Scaling factor for weights when replacing point neurons with multicompartment neurons
cfg.scaleConnWeight = 0.000001

cfg.simLabel = "pd_mc_scale-%s_DC-%d_TH-%d_Balanced-%d_dur-%d_wscale_%.6g" % (
    str(cfg.ScaleFactor),
    int(cfg.DC),
    int(cfg.TH),
    int(cfg.Balanced),
    int(cfg.duration / 1e3),
    cfg.scaleConnWeight,
)
"""
cfg.simLabel = f"SS_exc{cfg.excWeight}_inh{cfg.inhWeightScale}"

###########################################################
# Recording and plotting options
###########################################################

cfg.recordStep = 100  # 0.025  # Step size in ms to save data (e.g. V traces, LFP, etc)
cfg.filename = cfg.simLabel  # Set file output name
cfg.saveFolder = "dataSS4/"
cfg.savePickle = False  # Save params, network and sim output to pickle file
cfg.saveJson = True
cfg.saveDataInclude = ["simData", "simConfig"]
cfg.recordStim = False
cfg.printSynsAfterRule = False
# Dict with traces to record
# cfg.analysis['plotRaster'] = {'saveFig': True}                  # Plot a raster
# cfg.analysis['plotTraces'] = {'saveFig': True}  # Plot recorded traces for this list of cells


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

layer_bounds = {
    "L1": 0.08 * 1470,
    "L2": 0.27 * 1470,
    "L4": 0.58 * 1470,
    "L5": 0.73 * 1470,
    "L6": 1.0 * 1470,
}


# cfg.analysis['plotShape'] = {'includePost': cfg.recordCellsSpikes, 'includeAxon': 1, 'cvar': 'voltage', 'fontSize': 16, 'figSize': (12,8),
# 							'axis': 'on', 'axisLabels': False, 'saveFig': True, 'dpi': 300, 'dist': 0.65}

# plot 2D net structure
# cfg.analysis['plot2Dnet'] = {'include': cfg.recordCellsSpikes, 'saveFig': True,  'figSize': (10,15)}

# plot convergence connectivity as 2D
# cfg.analysis['plotConn'] = {'includePre': cfg.recordCellsSpikes, 'includePost': cfg.recordCellsSpikes, 'feature': 'convergence', \
#    'synOrConn': 'conn', 'graphType': 'bar', 'saveFig': True, 'figSize': (15, 9)}

# plot firing rate spectrogram  (run for 4 sec)
# cfg.analysis['plotRateSpectrogram'] = {'include': ['allCells'], 'saveFig': True, 'figSize': (15, 7)}

# plot granger causality (run for 4 sec)
# cfg.analysis.granger = {'cells1': ['L2i'], 'cells2': ['L4e'], 'label1': 'L2i', 'label2': 'L4e', 'timeRange': [500,4000], 'saveFig': True, 'binSize': 4}

# Scale synapses weights -- optimized with min K-leak 1e-5
cfg.excWeight_L2e = 0.034757230230507245
cfg.excWeight_L2i = 0.007435639178903054
cfg.excWeight_L4e = 0.00535800846139247
cfg.excWeight_L4i = 0.004635132492217301
cfg.excWeight_L5e = 0.0019321344065691371
cfg.excWeight_L5i = 0.006431462642063759
cfg.excWeight_L6e = 0.04045057917291838
cfg.excWeight_L6i = 0.005087528200972051

cfg.inhWeightScale_L2e = 5.4276668460112125
cfg.inhWeightScale_L2i = 8.756914769882469
cfg.inhWeightScale_L4e = 5.420514099517173
cfg.inhWeightScale_L4i = 7.495240884820139
cfg.inhWeightScale_L5e = 5.108836856739071
cfg.inhWeightScale_L5i = 4.90013157293792
cfg.inhWeightScale_L6e = 3.9141625146296994
cfg.inhWeightScale_L6i = 3.3945419748291674


cfg.gnabar = {
    "L2e": 0.013775199886666471,
    "L2i": 0.027571819187312125,
    "L4e": 0.02266830013423984,
    "L4i": 0.029102001912798815,
    "L5e": 0.02535744555783162,
    "L5i": 0.03081119693176582,
    "L6e": 0.014647818970212627,
    "L6i": 0.02446848239757097,
}
cfg.gkbar = {
    "L2e": 0.006256129398906307,
    "L2i": 0.006185825153046352,
    "L4e": 0.0060870611284361865,
    "L4i": 0.005724121639411816,
    "L5e": 0.003982061249246858,
    "L5i": 0.0055982570786592785,
    "L6e": 0.005668206201621326,
    "L6i": 0.005933783169921459,
}
cfg.ukcc2 = {
    "L2e": 0.0032046084164501472,
    "L2i": 0.0031509077144799,
    "L4e": 0.005288380837054354,
    "L4i": 0.00478331053234558,
    "L5e": 0.002703258989241554,
    "L5i": 0.010726802038549095,
    "L6e": 0.006728384451439758,
    "L6i": 0.005139646428462374,
}
cfg.unkcc1 = {
    "L2e": 2.9003078243583347,
    "L2i": 5.162627978681683,
    "L4e": 5.060143706759098,
    "L4i": 3.4421630020948286,
    "L5e": 7.306905091131788,
    "L5i": 1.6016563592554993,
    "L6e": 3.070483105176143,
    "L6i": 2.5419746302997566,
}
cfg.pmax = {
    "L2e": 5090.568389284457,
    "L2i": 4392.347665970014,
    "L4e": 7010.1014895806475,
    "L4i": 4076.953587990565,
    "L5e": 5258.713071981389,
    "L5i": 7070.669845884245,
    "L6e": 9419.798954366586,
    "L6i": 9050.513090329627,
}
cfg.gpas = {
    "L2e": 6.183693542550501e-05,
    "L2i": 2.6478580348538306e-05,
    "L4e": 2.474426208575386e-05,
    "L4i": 2.7062297576296835e-05,
    "L5e": 4.010647009384075e-05,
    "L5i": 5.412714996482607e-05,
    "L6e": 3.5297701447167655e-05,
    "L6i": 3.286565186740838e-05,
}

for k in ["gnabar", "gkbar", "ukcc2", "unkcc1", "pmax", "gpas"]:
    setattr(cfg, k, getattr(cfg, k)[cfg.popOpt[0]])
