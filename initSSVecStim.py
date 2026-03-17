"""
NetPyNE version of Potjans and Diesmann thalamocortical network

init.py -- code to run the simulation
"""

from netpyne import sim
from neuron import h, rxd
import pandas as pd
import numpy as np
import pickle
import sys
import traceback
############################################################
#               Create network and run simulation
############################################################


def fi(cells):
    """set steady state RMP for 1 cell"""
    for c in cells:
        # skip artificial cells
        if not hasattr(c.secs, "soma"):
            continue
        seg = c.secs.soma.hObj(0.5)
        isum = 0
        isum = (
            (seg.ina if h.ismembrane("na_ion") else 0)
            + (seg.ik if h.ismembrane("k_ion") else 0)
            + (seg.ica if h.ismembrane("ca_ion") else 0)
            + (seg.iother if h.ismembrane("other_ion") else 0)
        )
        seg.e_pas = cfg.hParams["v_init"] + isum / seg.g_pas
        if h.ismembrane("cadad"):
            seg.cainf_cadad = seg.cai - (
                (-(10000) * seg.ica / (2 * h.FARADAY * seg.depth_cadad))
                * seg.taur_cadad
            )

depolarized = []
twice_depolarized = []
def runFunc(t):
    global depolarized, twice_depolarized

    # give up after 200 ms if no APs
    if t>=200 and len(sim.simData["spkid"]) == 0:
        print("No spikes detected after 100 ms, stopping simulation.")
        h.t = sim.cfg.duration

    # give up after >= 300ms if a cell is >-10 mV for 3 time checks in a row
    for cell in sim.net.cells:
        if cell.tags['cellModel'] != "VecStim" and cell.tags['cellModel'] != "NetStim":
            if cell.secs['soma']['hObj'].v > -10:
                if cell.gid not in depolarized:
                    depolarized.append(cell.gid)
                elif cell.gid not in twice_depolarized:
                    twice_depolarized.append(cell.gid)
                else:
                    print(f"Cell {cell.gid} is depolarized above -10 mV for an extended period, stopping simulation.")
                    h.t = sim.cfg.duration
            else:
                if cell.gid in depolarized:
                    depolarized.remove(cell.gid)
                if cell.gid in twice_depolarized:
                    twice_depolarized.remove(cell.gid)

try:
    simConfig, netParams = sim.readCmdLineArgs(
        simConfigDefault="cfgSS.py", netParamsDefault="netParamsSSVecStim.py"
    )
    sim.initialize(
        simConfig=simConfig, netParams=netParams
    )  # create network object and set cfg and net params
    sim.net.createPops()  # instantiate network populations
    sim.net.createCells()  # instantiate network cells based on defined populations
    sim.net.addStims()  # add network stimulation
    # fih = h.FInitializeHandler(2, lambda: fi(sim.net.cells))
    sim.net.addRxD(nthreads=2)
    
    clamps = []
    for cell in sim.net.cells:
        if cell.tags['cellModel'] != "VecStim" and cell.tags['cellModel'] != "NetStim":
            vclamp = h.VClamp(cell.secs['soma']['hObj'](0.5))
            vclamp.dur[0] = 25
            vclamp.dur[1] = 0
            vclamp.dur[2] = 0
            vclamp.amp[0] = -70
            clamps.append(vclamp)
    
    
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
    
    # extra recording
    """
    for sp in rxd.species._all_defined_species:
        if sp().name == 'mgate':
            mgate = sp()
        elif sp().name == 'hgate':
            hgate = sp()
        elif sp().name == 'ngate':
            ngate = sp()
    extraRec = {}
    for cellName in sim.cfg.recordCells:
        dat = {}
        cell = sim.getCellsList(include=[cellName])[0]
        dat['mgate'] = h.Vector().record(mgate.nodes(cell.secs['soma']['hObj'])._ref_value, sim.cfg.recordStep)
        dat['hgate'] = h.Vector().record(hgate.nodes(cell.secs['soma']['hObj'])._ref_value, sim.cfg.recordStep)
        dat['ngate'] = h.Vector().record(ngate.nodes(cell.secs['soma']['hObj'])._ref_value, sim.cfg.recordStep)
        extraRec[cellName] = dat
    """ 
    sim.runSimWithIntervalFunc(100, runFunc)

    sim.gatherData()  # gather spiking data and cell info from each node
    sim.saveData()  # save params, cell info and sim output to file (pickle,mat,txt,etc)#

except Exception:
    print("Exception occurred!")
    traceback.print_exc()
finally:
    print("Calling h.quit()")
    h.quit()


# sim.analysis.plotData()               # plot spike raster etc

#pickle.dump(extraRec, open(f"{sim.cfg.saveFolder}/extraRec.pkl",'wb'))

# # Plot all electrodes separately; use electrode 6
# for elec in [3]: #range(15):
# 	sim.analysis.plotLFP(**{'plots': ['PSD'], 'electrodes': [elec], 'timeRange': [100,600], 'maxFreq':80, 'figSize': (7,4), 'fontSize': 16, 'saveData': False, 'saveFig': cfg.saveFolder+cfg.simLabel+'_LFP_PSD_elec_'+str(elec)+'.png', 'showFig': False})
# 	#sim.analysis.plotLFP(**{'plots': ['spectrogram'], 'electrodes': [elec], 'timeRange': [100,600], 'maxFreq':80, 'figSize': (8,4), 'fontSize': 16, 'saveData': False, 'saveFig': cfg.saveFolder+cfg.simLabel+'_LFP_spec_elec_'+str(elec)+'.png', 'showFig': False})
