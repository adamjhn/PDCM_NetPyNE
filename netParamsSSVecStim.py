"""
NetPyNE version of single cells from the Potjans and Diesmann thalamocortical network with multicompartment neurons

netParams.py -- contains the network parameters (netParams object)

Modified to include concentration of Na, K, Cl and O2 using RxD.

"""

from netpyne import specs
import numpy as np
from neuron.units import sec, mM, M, s, um
import math
import json
import pickle
import re

def rand_uniform(gid=0):
    r = h.Random()
    r.Random123(gid, 1, 1)
    return r.uniform(-75, -60)


try:
    from __main__ import cfg  # import SimConfig object with params from parent module
except:
    from cfgSS import (
        cfg,
    )  # if no simConfig in parent module, import directly from cfg.py:cfg


# examples of  input/output relation for LIF network model
data = pickle.load(
    open(
        "sample_pd_scale-0.16_DC-0_TH-1_Balanced-1_dur-1.pkl",
        "rb",
    )
)
cfg.data = data


############################################################
#
#                    NETWORK PARAMETERS
#
############################################################

# Population size N
L = list(data.keys())
N_Full = np.array([len(data[pop]["cellGids"]) for pop in L])

############################################################
# NetPyNE Network Parameters (netParams)
############################################################

netParams = (
    specs.NetParams()
)  # object of class NetParams to store the network parameters

############################################################
# Populations parameters
############################################################

# population locations
# from Schmidt et al 2018, PLoS Comp Bio, Macaque V1
netParams.sizeX = cfg.sizeX  # x-dimension (horizontal length) size in um
netParams.sizeY = cfg.sizeY  # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = cfg.sizeZ  # z-dimension (horizontal depth) size in um
netParams.shape = "cylinder"  # cylindrical (column-like) volume

popDepths = [
    [0.08, 0.27],
    [0.08, 0.27],
    [0.27, 0.58],
    [0.27, 0.58],
    [0.58, 0.73],
    [0.58, 0.73],
    [0.73, 1.0],
    [0.73, 1.0],
]


# cell property rules -- single compartment model from population SD model
for pop in L:
    cellRule = netParams.importCellParams(
        label="cellRule",
        fileName="Neuron.py",
        conds={"cellType": "SC", "cellModel": pop},
        cellName=pop,
    )
    netParams.cellParams[f"{pop}"] = cellRule


############################################################
## Synaptic mechanism parameters
############################################################
netParams.synMechParams["exc"] = {
    "mod": "AMPA",
    "tau1": 0.8,
    "tau2": 5.3,
}  # AMPA synaptic mechanism
netParams.synMechParams["inh"] = {
    "mod": "GABA",
    "tau1": 0.6,
    "tau2": 8.5,
}  # GABA synaptic mechanism


############################################################
# External input parameters
############################################################
# Reescaling function (move to separate module?)
def Reescale(ScaleFactor, C, N_Full, w_p, f_ext, tau_syn, Inp, InpDC):
    if ScaleFactor < 1.0:
        # This is a good approximation of the F_out param for the Balanced option "True".
        # Note for the Balanced=False option, it should be possible to calculate a better approximation.
        F_out = np.array([0.860, 2.600, 4.306, 5.396, 8.142, 8.188, 0.941, 7.3])

        Ncon = np.vstack(
            [np.column_stack([0 for i in range(0, 8)]) for i in range(0, 8)]
        )
        for r in range(0, 8):
            for c in range(0, 8):
                Ncon[r][c] = (
                    np.log(1.0 - C[r][c]) / np.log(1.0 - 1.0 / (N_Full[r] * N_Full[c]))
                ) / N_Full[r]

        w = w_p * np.column_stack(
            [np.vstack([[1.0, -4.0] for i in range(0, 8)]) for i in range(0, 4)]
        )
        w[0][2] = 2.0 * w[0][2]

        x1_all = w * Ncon * F_out
        x1_sum = np.sum(x1_all, axis=1)
        x1_ext = w_p * Inp * f_ext
        I_ext = np.column_stack([0 for i in range(0, 8)])
        I_ext = (
            0.001
            * tau_syn
            * (
                (1.0 - np.sqrt(ScaleFactor)) * x1_sum
                + (1.0 - np.sqrt(ScaleFactor)) * x1_ext
            )
        )

        InpDC = np.sqrt(ScaleFactor) * InpDC * w_p * f_ext * tau_syn * 0.001  # pA
        w_p = w_p / np.sqrt(ScaleFactor)  # pA
        InpDC = InpDC + I_ext
        N_ = [max(int(ScaleFactor * N), 1) for N in N_Full]
    else:
        InpDC = InpDC * w_p * f_ext * tau_syn * 0.001
        N_ = N_Full

    return InpDC, N_, w_p


###########################################################
#  Network Constants
###########################################################

# Frequency of external input
f_ext = 8.0  # (Hz)
# Postsynaptic current time constant
tau_syn = 0.5  # (ms)
# Membrane time constant
tau_m = 10  # (ms)
# Other constants defined inside .mod
"""
#Absolute refractory period
tauref =2 (ms) 
#Reset potential
Vreset = -65 (mV) : -49 (mV) :
#Fixed firing threshold
Vteta  = -50 (mV)"""
# Membrane capacity
C_m = cfg.Cm / (2e-8 * np.pi * cfg.somaR ** 2)  # pF
# Mean amplitude of the postsynaptic potential (in mV).
w_v = 0.15
# Mean amplitude of the postsynaptic potential (in pA).
w_p = (
    (C_m) ** (-1)
    * tau_m
    * tau_syn
    / (tau_syn - tau_m)
    * (
        (tau_m / tau_syn) ** (-tau_m / (tau_m - tau_syn))
        - (tau_m / tau_syn) ** (-tau_syn / (tau_m - tau_syn))
    )
) ** (-1)
w_p = w_v * w_p  # (pA)

# C probability of connection
C = np.array(
    [
        [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
        [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
        [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
        [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
        [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
        [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
        [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
        [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
    ]
)

# Population size N
if cfg.singleCells:
    N_Full = np.array([1 for _ in L])

# Number of Input per Layer
Inp = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
if cfg.Balanced == False:
    InpUnb = np.array([2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850])
###########################################################
# Reescaling calculation
###########################################################
if cfg.DC == True:
    InpDC = Inp
    if cfg.Balanced == False:
        InpDC = InpUnb
else:
    InpDC = np.zeros(8)
    InpPoiss = Inp * cfg.ScaleFactor
    if cfg.Balanced == False:
        InpPoiss = InpUnb * cfg.ScaleFactor
N_ = N_Full

############################################################
# NetPyNE Network Parameters (netParams)
############################################################

netParams.delayMin_e = 1.5
netParams.ddelay = 0.5
netParams.delayMin_i = 0.75
netParams.weightMin = cfg.weightMin
netParams.dweight = cfg.dWeight 

# cell property rules
for pop in L:
    cellRule = netParams.importCellParams(
        label="cellRule",
        fileName="Neuron.py",
        conds={"cellType": pop, "cellModel": pop},
        cellName=pop,
    )
    netParams.cellParams[pop + "Rule"] = cellRule

# ------------------------------------------------------------------------------
# create populations
all_cells = []
for i in range(0, 8):
    for idx in range(N_[i]):
        if (i == 0) or (i == 1):
            netParams.popParams[f"{L[i]}_{idx}"] = {
                "cellType": str(L[i]),
                "numCells": 1,
                "cellModel": L[i],
                "xRange": [0.0, cfg.sizeX],
                "yRange": [popDepths[i][0] * cfg.sizeY, cfg.sizeY * popDepths[i][1]],
                "zRange": [0.0, cfg.sizeZ],
            }
        elif (i == 6) or (i == 7):
            netParams.popParams[f"{L[i]}_{idx}"] = {
                "cellType": str(L[i]),
                "numCells": 1,
                "cellModel": L[i],
                "xRange": [0.0, cfg.sizeX],
                "yRange": [
                    popDepths[i][0] * cfg.sizeY,
                    cfg.sizeY * popDepths[i][1] - 2 * cfg.somaR,
                ],
                "zRange": [0.0, cfg.sizeZ],
            }
        else:
            netParams.popParams[f"{L[i]}_{idx}"] = {
                "cellType": str(L[i]),
                "numCells": 1,
                "cellModel": L[i],
                "xRange": [0.0, cfg.sizeX],
                "yRange": [popDepths[i][0] * cfg.sizeY, cfg.sizeY * popDepths[i][1]],
                "zRange": [0.0, cfg.sizeZ],
            }
        all_cells.append(f"{L[i]}_{idx}")

############################################################
# Connectivity parameters
############################################################
def filterTimes(inputs, weights, offset=-1, thresh=1e-9):
    """ sum inputs that are less than `thresh` apart and shift by `offset`
        The offset allows for non-zero delay when replaying the inputs.
    """ 
    inp = [max(0,inputs[0] + offset)]   # start with the first input
    wei = [weights[0]]
    for t, w in zip(inputs[1:], weights[1:]):
        tnext = max(0,t + offset)       # time of next input
        if tnext - inp[-1] > thresh:
            inp.append(tnext)           # add next input
            wei.append(w)
        else:
            wei[-1] += w        # keep current input -- but update weight
        # stop if input time exceeds duration
        if inp[-1] >= cfg.duration:
            break
    return (inp, wei)


# cells are disconnect and just receive pre-recorded network activity
for pop, sz in zip(L, N_Full):
    for idx in range(sz):
        inp = data[pop]["inputs"][idx]
        wei = data[pop]["weight"][idx]
        exc = wei >= 0
        inh = wei < 0
        # sum inputs less than 10^-12s apart
        for syn in ["exc", "inh"]:
            if syn == "exc":
                inputs, weights = filterTimes(inp[exc], wei[exc])
                weightScale = getattr(cfg, f"excWeight_{pop}", cfg.excWeight)
            else:
                inputs, weights = filterTimes(inp[inh], wei[inh])
                excW = getattr(cfg, f"excWeight_{pop}", cfg.excWeight)
                inhS = getattr(cfg, f"inhWeightScale_{pop}", cfg.inhWeightScale)
                weightScale = excW * inhS
            # create a source from the pre-recorded spikes
            netParams.popParams[f"vec_{pop}_{idx}_{syn}"] = {
                "cellModel": "VecStim",
                "type": "VecStim",
                "spkTimes": inputs,
                "weights": weights,
                "numCells": 1,
            }
            # play the source into the target cell
            netParams.connParams[f"conn_{pop}_{idx}_{syn}"] = {
                "preConds": {"pop": f"vec_{pop}_{idx}_{syn}"},
                "postConds": {"pop": f"{pop}_{idx}"},
                'probability': 1.0,
                "weight": weightScale,  # synaptic weight
                "delay": 1,
                "synMech": syn,
            }

############################################################
# RxD params
############################################################

### constants
e_charge = 1.60217662e-19
scale = 1e-14 / e_charge
alpha = 5.3  # g/mol
dx = cfg.dx

constants = {
    "e_charge": e_charge,
    "scale": scale,
    "gnabar": cfg.gnabar * scale,  # molecules/um2 ms mV ,
    "gnabar_l": (0.0247 / 1000) * scale,
    "gkbar": cfg.gkbar * scale,
    "gkbar_l": (0.05 / 1000) * scale,
    "gclbar_l": (0.1 / 1000) * scale,
    "ukcc2": cfg.ukcc2 * mM / sec,
    "unkcc1": cfg.unkcc1 * mM / sec,
    "alpha": alpha,
    "epsilon_k_max": 0.25 / sec,
    "epsilon_o2": 0.17 / sec,
    "vtau": 1 / 250.0,
    "g_gliamax": 5 * mM / sec,
    "beta0": 7.0,
    "avo": 6.0221409 * (10 ** 23),
    "p_max": cfg.pmax /um**2,
    "nao_initial": 144.0,
    "nai_initial": 18.0,
    "gnai_initial": 18.0,
    "gki_initial": 80.0,
    "ko_initial": 3.5,
    "ki_initial": 140.0,
    "clo_initial": 130.0,
    "cli_initial": 6.0,
    "ATP_initial": cfg.ATPss,
    "ADP_initial": cfg.ADPss,
    "AMP_initial": cfg.AMPss,
    "Pos_initial": cfg.Pss,
    "o2_bath": cfg.o2_bath,
    "o2_init": cfg.o2_init,
    "v_initial": cfg.hParams["v_init"],
}

# Pump model

# ADP[cyt] is the total ADP (D2 is MgADP, D1 is free ADP) 
AKb = f"{1/cfg.AK['KMg']} + ADP[cyt] + {cfg.Mg}" 
D2 = f"({AKb} - (({AKb})**2 - ADP[cyt] * {4*cfg.Mg})**0.5)/2"
D1 = f"(ADP[cyt] - ({D2}))"


q10 = 3.2
F=96485.33212331001
R=8.31446261815324
T=cfg.hParams["celsius"] + 273.15
Tref = 310
q10_factor = q10**((T - Tref)/10.0)
Delta = -0.031
# parameters
k1p = 1050/s * q10_factor
k1m = 172.1/s/mM * q10_factor
k2p = 481/s * q10_factor
k2m = 40/s * q10_factor
k3p = 2000/s * q10_factor
k3m = 79.3e3/s/mM**2 * q10_factor
k4p = 320/s * q10_factor
k4m = 40/s * q10_factor
KATP = 2.51*mM 
KHPi = 6.77*mM
KKPi = 292*mM
KNaPi = 224*mM
#MgATP = 9.8
#MgADP = 0.05
pH = 7.0
PiT = 4.2*mM

KKe = 0.213*mM
KKi = 0.5*mM
KNae = f"({15.5*mM} * rxd.rxdmath.exp({(1+Delta)*F*1e-3/(3*R*T)}*rxd.v))"
KNai = f"({2.49*mM} * rxd.rxdmath.exp({Delta*F*1e-3/(3*R*T)}*rxd.v))"

# equations
Nai = f"(nai[cyt]/{KNai})"
Nae = f"(nao[ecs]/{KNae})"
Ki = f"(kki[cyt]/{KKi})"
Ke = f"(kko[ecs]/{KKe})"
eATP = f"ATP[cyt]/{KATP}"
H = 10**(-pH)*M 
Pi = f"{PiT}/(kki[cyt]/{KKPi} + {1 + H/KHPi} + nai[cyt]/{KNaPi})"
a1p = f"{k1p}*{Nai}**3/((1+{Nai})**3 + (1+{Ki})**2 - 1)"    #/ms
a2p = k2p                                                   #/ms
a3p = f"{k3p}*{Ke}**2/((1+{Nae})**3 + (1+{Ke})**2 - 1)"     #/ms    
a4p = f"{k4p}*{eATP}/(1+{eATP})"                            #/ms

a1m = f"{k1m}*({D2})"                                       #/ms
a2m = f"{k2m}*{Nae}**3/((1+{Nae})**3 + (1+{Ke})**2 - 1)"    #/ms
a3m = f"{k3m}*{Pi}*{H}/(1+{eATP})"                          #/ms
a4m = f"{k4m}*{Ki}**2/((1+{Nai})**3 + (1+{Ki})**2 - 1)"     #/ms
Sigma = f"""{a3m}*{a2m}*{a1m} +
            {a4p}*{a2m}*{a1m} +
            {a4p}*{a2p}*{a3p} +
            {a4p}*{a1m}*{a3p} +
            {a3m}*{a2m}*{a1p} +
            {a4p}*{a2m}*{a1p} + 
            {a4p}*{a1p}*{a3p} +
            {a3m}*{a1p}*{a2p} + 
            {a4p}*{a1p}*{a2p} +
            {a1p}*{a2p}*{a3p} +
            {a4m}*{a3m}*{a1m} + 
            {a4m}*{a3m}*{a2p} +
            {a4m}*{a3m}*{a2m} + 
            {a4m}*{a1m}*{a2m} +
            {a4m}*{a2p}*{a3p} +
            {a4m}*{a1m}*{a3p}"""        #/ms**3
Sigma = re.sub(r'\s+', ' ', Sigma)

# each pump cycle 
# ATP + 3Nai + 2Ke <> ADP + Pi + 3Nae + 2Ki
# Here ATP is produced by consuming O2, ADP and Pi
#      ADP is restored towards a SS concentration
#      P   is restored towards a SS concetration
pumpRate = f"p_max * ({a1p}*{a2p}*{a3p}*{a4p} -{a1m}*{a2m}*{a3m}*{a4m})/({Sigma})"      #/ms

# sodium activation 'm'
alpha_m = "(0.32 * (rxd.v + 54.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 54.0)/4.0))"
beta_m = "(0.28 * (rxd.v + 27.0))/(rxd.rxdmath.exp((rxd.v + 27.0)/5.0) - 1.0)"
alpha_m0 = (0.32 * (constants["v_initial"] + 54.0)) / (
    1.0 - math.exp(-(constants["v_initial"] + 54) / 4.0)
)
beta_m0 = (0.28 * (constants["v_initial"] + 27.0)) / (
    math.exp((constants["v_initial"] + 27.0) / 5.0) - 1.0
)
m_initial = alpha_m0 / (beta_m0 + alpha_m0)

# sodium inactivation 'h'
alpha_h = "0.128 * rxd.rxdmath.exp(-(rxd.v + 50.0)/18.0)"
beta_h = "4.0/(1.0 + rxd.rxdmath.exp(-(rxd.v + 27.0)/5.0))"
alpha_h0 = 0.128 * math.exp(-(constants["v_initial"] + 50.0) / 18.0)
beta_h0 = 4.0 / (1.0 + math.exp(-(constants["v_initial"] + 27.0) / 5.0))
h_initial = alpha_h0 / (beta_h0 + alpha_h0)

# potassium activation 'n'
alpha_n = "(0.032 * (rxd.v + 52.0))/(1.0 - rxd.rxdmath.exp(-(rxd.v + 52.0)/5.0))"
beta_n = "0.5 * rxd.rxdmath.exp(-(rxd.v + 57.0)/40.0)"
alpha_n0 = (0.032 * (constants["v_initial"] + 52.0)) / (
    1.0 - math.exp(-(constants["v_initial"] + 52.0) / 5.0)
)
beta_n0 = 0.5 * math.exp(-(constants["v_initial"] + 57.0) / 40.0)
n_initial = alpha_n0 / (beta_n0 + alpha_n0)


### reactions
gna = "gnabar*mgate**3*hgate"
gk = "gkbar*ngate**4"
fko = "1.0 / (1.0 + rxd.rxdmath.exp(16.0 - kko[ecs] / vol_ratio[ecs]))"
nkcc1A = "rxd.rxdmath.log((kki[cyt] * cli[cyt] / vol_ratio[cyt]**2) / (kko[ecs] * clo[ecs] / vol_ratio[ecs]**2))"
nkcc1B = "rxd.rxdmath.log((nai[cyt] * cli[cyt] / vol_ratio[cyt]**2) / (nao[ecs] * clo[ecs] / vol_ratio[ecs]**2))"
nkcc1 = f"(unkcc1 * ({fko}) * ({nkcc1A} + {nkcc1B}))"
kcc2 = "(ukcc2 * rxd.rxdmath.log((kki[cyt] * cli[cyt] * vol_ratio[cyt]**2) / (kko[ecs] * clo[ecs] * vol_ratio[ecs]**2)))"

# Nerst equation - reversal potentials
ena = "26.64 * rxd.rxdmath.log(nao[ecs]*vol_ratio[cyt]/(nai[cyt]*vol_ratio[ecs]))"
ek = "26.64 * rxd.rxdmath.log(kko[ecs]*vol_ratio[cyt]/(kki[cyt]*vol_ratio[ecs]))"
ecl = "26.64 * rxd.rxdmath.log(cli[cyt]*vol_ratio[ecs]/(clo[ecs]*vol_ratio[cyt]))"

o2ecs = "oxygen[ecs]"
o2cyt = "oxygeni[cyt]"
# Wei model has o2 baseline 32mg/L, i.e. varies between 0 and 6.0mM
# Our model has o2 baseline of 0.06mM bath or 0.04mM initial
# to provide a similar sigmoid curve for the range of oxygen considered
# currents were scaled by 32/0.05 = 640 mg/L/mM
rescale_o2 = 32 * 20

#switches used to avoid concentration becoming negative
o2switch = "((1.0 + rxd.rxdmath.tanh(1e5 * (%s - 5e-4))) / 2.0)" % (o2cyt)
ATPswitch = "((1.0 + rxd.rxdmath.tanh(1e5 * (ATP[cyt] - 5e-4))) / 2.0)"
ADPswitch = "((1.0 + rxd.rxdmath.tanh(1e5 * (ADP[cyt] - 5e-4))) / 2.0)"
AMPswitch = "((1.0 + rxd.rxdmath.tanh(1e5 * (AMP[cyt] - 5e-4))) / 2.0)"
PosSwitch = "((1.0 + rxd.rxdmath.tanh(1e5 * (Pos[cyt] - 5e-4))) / 2.0)"

avo = 6.0221409 * (10 ** 23)
volume_scale = 1e-18 * avo / cfg.sa2v # convert mM/ms -> mol/um**2/ms
osm = "(1.1029 - 0.1029*rxd.rxdmath.exp( ( (nao[ecs] + kko[ecs] + clo[ecs] + 18.0)/vol_ratio[ecs] - (nai[cyt] + kki[cyt] + cli[cyt] + cli[cyt] + 132.0)/vol_ratio[cyt])/20.0))"
scalei = str(avo * 1e-18)
scaleo = str(avo * 1e-18)

# ATP synthase kinetics with Michaelis-Menten terms for all substrates
# Km values: PMC3833997 (KmADP=0.025mM), PMC8434986 (KmPi=1.0mM), Ko2=0.3μM (cytochrome c oxidase)
# Physiological ATP turnover: 0.516 uM/ms in grey matter (PMC12415541), 0.015 mM/s basal
Vatp  = f"{o2switch} * {ADPswitch} * {PosSwitch} * "
Vatp += f"(ADP[cyt]/(ADP[cyt] + {cfg.KmADP_synthase})) * "
Vatp += f"(Pos[cyt]/(Pos[cyt] + {cfg.KmPi_synthase})) * "
Vatp += f"({o2cyt}/({cfg.Ko2} + {o2cyt})) * "
Vatp += f"({cfg.KiATP_synthase}/({cfg.KiATP_synthase} + ATP[cyt]))"


# update constants to ensure net zero flux at RMP
evalInit = {
    "vol_ratio[ecs]": "1.0",
    "vol_ratio[cyt]": "1.0",
    "rxd.rxdmath": "math",
    "rxd.v": constants["v_initial"],
    "kki[cyt]": constants["ki_initial"],
    "kko[ecs]": constants["ko_initial"],
    "nai[cyt]": constants["nai_initial"],
    "nao[ecs]": constants["nao_initial"],
    "cli[cyt]": constants["cli_initial"],
    "clo[ecs]": constants["clo_initial"],
    "ATP[cyt]": constants["ATP_initial"],
    "ADP[cyt]": constants["ADP_initial"],
    "AMP[cyt]": constants["AMP_initial"],
    "Pos[cyt]": constants["Pos_initial"],
    "oxygen[ecs]": constants["o2_init"],
    "oxygeni[cyt]": constants["o2_init"],
    "ngate": n_initial,
    "mgate": m_initial,
    "hgate": h_initial,
}


def initEval(ratestr):
    for k, v in evalInit.items():
        ratestr = ratestr.replace(k, str(v))
    for k, v in constants.items():
        ratestr = ratestr.replace(k, str(v))
    return eval(ratestr)
# check pump can balance K+ currents at rest with min leak 1e-5 mS/cm^2
min_pmax = f"p_max * ({volume_scale}*({nkcc1} + {kcc2}) + {gk} * (v_initial - {ek}))/(2*{pumpRate})"
min_leak = initEval(f"1e-5*p_max*{scale}*(v_initial - {ek})/((2*{pumpRate}))")
pmin = initEval(min_pmax)
if constants["p_max"] < pmin + min_leak:
    print("Pump current is too low to balance K+ currents with gkbar_l 5e-5")
    print(f"p_max set to {pmin + min_leak}")
    constants["p_max"] = pmin + min_leak


# scale by area(um**2)/(N_A * volume(m**3) * vol_fraction) to give cyt mM/ms
Prescale = 4*np.pi*cfg.somaR**2/(1e-18*avo*2*np.pi*cfg.somaR**3*cfg.cyt_fraction)
pr = initEval(pumpRate)*Prescale # cyt mM/ms ATP->ADP + Pos
print(f"Pump ATP consumption = {pr:.6e} mM/ms")
print(f"Basal ATP consumption = {pr/2:.6e} mM/ms")
print(f"Total ATP consumption = {1.5*pr:.6e} mM/ms")

VBasalATP = pr/2
Vmax_ATPsyth = (1.5*pr)/(5*initEval(Vatp))
print(f"Vmax_ATPsynthase = {Vmax_ATPsyth:.6e} mM/ms")
Vatp_rate = 5*Vmax_ATPsyth*initEval(Vatp)
print(f"ATP production rate = {Vatp_rate:.6e} mM/ms")

# forward rate
MTdivKTKM = f"(AMP[cyt] * ATP[cyt]/{cfg.AK['KmAMP'] * cfg.AK['KmMgATP']})"
fnum = f"({cfg.AK['kp2']*cfg.AK['km1']} * {MTdivKTKM})"
fden = f"({cfg.AK['km1']}*(1 + {1/cfg.AK['KmAMP'] + 1/cfg.AK['KiAMP']}*AMP[cyt] + ATP[cyt]/{cfg.AK['KmMgATP']} + {MTdivKTKM}) + {cfg.AK['kp1'] + cfg.AK['kp2']} * {MTdivKTKM})"

# backward rate
DDdivKDKD = f"(({D1})*({D2}) /({cfg.AK['KmADP']*cfg.AK['KmMgADP']}))"
bnum = f"({cfg.AK['km2'] * cfg.AK['km1']} * {DDdivKDKD})"
bden = f"({cfg.AK['km1']} + {cfg.AK['km2']} * {DDdivKDKD} + {cfg.AK['kp1']}*(1 + {1/cfg.AK['KmADP'] + 1/cfg.AK['KmMgADP']}*{D1} + {D2}/{cfg.AK['KmMgADP']} + {DDdivKDKD}))"

ADK_forward_rate = initEval(f"{fnum}/{fden}")

ADK_backward_rate = initEval(f"{bnum}/{bden}")

print(f"\nAdenylate kinase at steady state:")
print(f"  Forward rate (2*ADP → ATP+AMP) = {ADK_forward_rate:.6e} mM/ms")
print(f"  Backward rate (ATP+AMP → 2*ADP) = {ADK_backward_rate:.6e} mM/ms")
print(f"  Net rate = {ADK_forward_rate - ADK_backward_rate:.6e} mM/ms")
print(f"  Equilibrium constant Keq = [ATP][AMP]/[ADP]² = {constants['ATP_initial']*constants['AMP_initial']/constants['ADP_initial']**2:.3f}")

# Verify steady state balance
print(f"\nSteady-state verification:")
print(f"  ATP production - consumption = {Vatp_rate - 1.5*pr:.6e} mM/ms (should be ~0)")
print(f"  Pi production - consumption = {1.5*pr - Vatp_rate:.6e} mM/ms (should be ~0)")

Vatp += "*" + str(Vmax_ATPsyth)


# rescale pmax
"""
pA = "(1.0 / (1.0 + rxd.rxdmath.exp((25.0 - nai[cyt] / vol_ratio[cyt])/3.0)))"
pB = "(1.0 / (1.0 + rxd.rxdmath.exp(3.5 - kko[ecs] / vol_ratio[ecs])))"
rA = f"(1.0 / (1.0 + rxd.rxdmath.exp(({cfg.KNai} - nai[cyt] / vol_ratio[cyt])/3.0)))"
rB = f"(1.0 / (1.0 + rxd.rxdmath.exp({cfg.KKo} - kko[ecs] / vol_ratio[ecs])))"
rescale = initEval(f"{pA} * {pB}/({rA} * {rB})")
constants["p_max"] = constants["p_max"] * rescale
"""
clbalance = f"(-(2*{nkcc1} + {kcc2}) * {volume_scale})/({ecl} - v_initial)"
kbalance = f"({gk} * (v_initial - {ek}) + ({volume_scale} * ({nkcc1} + {kcc2})  -2.0 * {pumpRate}))  / ({ek} - v_initial)"
nabalance = f"({gna} * (v_initial - {ena}) + ({nkcc1}*{volume_scale} + 3.0 * {pumpRate})) / ({ena} - v_initial)"

Kchan0 = initEval(f"{gk} * (v_initial - {ek})")
Knkcc10 = initEval(f"{volume_scale} * {nkcc1}")
Kkcc20 = initEval(f"{volume_scale} * {kcc2}")
Knakpump0 = -2*initEval(pumpRate)
Kleak0 = initEval(kbalance)*initEval(f"(v_initial - {ek})")
Ktot = Kchan0 + Knkcc10 + Kkcc20 + Knakpump0 + Kleak0


constants["gclbar_l"] = initEval(clbalance)
constants["gkbar_l"] = cfg.gkleak_scale * initEval(kbalance)
constants["gnabar_l"] = initEval(nabalance)

if constants["gkbar_l"] < 0:
    if abs(constants["gkbar_l"])< 1e-9:
        constants["gkbar_l"] = 0
    else:
        raise Exception(f"Negative leak gkbar_l: {constants['gkbar_l']}")
netParams.rxdParams["constants"] = constants

### regions
regions = {}

#### ecs dimensions
# margin = cfg.somaR
x = [0, cfg.sizeX]
y = [-cfg.sizeY, 0]
z = [0, cfg.sizeZ]

regions["ecs"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": dx,
    "volume_fraction": cfg.alpha_ecs,
    "tortuosity": cfg.tort_ecs,
}

regions["cyt"] = {
    "cells": all_cells,
    "secs": "all",
    "nrn_region": "i",
    "geometry": {
        "class": "FractionalVolume",
        "args": {"volume_fraction": cfg.cyt_fraction, "surface_fraction": 1},
    },
}

regions["mem"] = {
    "cells": all_cells,
    "secs": "all",
    "nrn_region": None,
    "geometry": "membrane",
}

netParams.rxdParams["regions"] = regions

### species
species = {}

species["kki"] = {
    "regions": ["cyt"],
    "d": 2.62,
    "charge": 1,
    "initial": "ki_initial",
    "name": "k",
}

species["nai"] = {
    "regions": ["cyt"],
    "d": 1.78,
    "charge": 1,
    "initial": "nai_initial",
    "name": "na",
}

species["cli"] = {
    "regions": ["cyt"],
    "d": 2.1,
    "charge": -1,
    "initial": "cli_initial",
    "name": "cl",
}


netParams.rxdParams["species"] = species

### parameters
params = {}
params["oxygen"] = {
    "regions": ["ecs"],
    "value": constants["o2_init"],
}  # constants['o2_bath']}
params["kko"] = {
    "regions": ["ecs"],
    "charge": 1,
    "name": "k",
    "value": constants["ko_initial"],
}
params["nao"] = {
    "regions": ["ecs"],
    "charge": 1,
    "name": "na",
    "value": constants["nao_initial"],
}

params["clo"] = {
    "regions": ["ecs"],
    "charge": -1,
    "name": "cl",
    "value": constants["clo_initial"],
}

params["dump"] = {"regions": ["cyt", "ecs"], "name": "dump"}


netParams.rxdParams["parameters"] = params

### states
netParams.rxdParams["states"] = {
    "vol_ratio": {"regions": ["cyt", "ecs"], "initial": 1.0, "name": "volume"},
    "mgate": {"regions": ["mem"], "initial": m_initial, "name": "mgate"},
    "hgate": {"regions": ["mem"], "initial": h_initial, "name": "hgate"},
    "ngate": {"regions": ["mem"], "initial": n_initial, "name": "ngate"},
    "o2_consumed": {"regions": ["ecs"], "initial": 0, "name": "o2_consumed"},
    "ATP": {"regions":["cyt"], "initial": constants["ATP_initial"], "name": "ATP"},
    "ADP": {"regions":["cyt"], "initial": constants["ADP_initial"], "name": "ADP"},
    "AMP": {"regions":["cyt"], "initial": constants["AMP_initial"], "name": "AMP"},
    "Pos": {"regions":["cyt"], "initial": constants["Pos_initial"], "name": "Pos"},
    "oxygeni": {"regions":["cyt"], "initial": constants["o2_init"], "name": "oxygeni"},
}

### reactions
mcReactions = {}

## volume dynamics
mcReactions["vol_dyn"] = {
    "reactant": "vol_ratio[cyt]",
    "product": "dump[ecs]",
    "rate_f": "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scalei, osm),
    "membrane": "mem",
    "custom_dynamics": True,
    "scale_by_area": False,
}

mcReactions["vol_dyn_ecs"] = {
    "reactant": "dump[cyt]",
    "product": "vol_ratio[ecs]",
    "rate_f": "-1 * (%s) * vtau * ((%s) - vol_ratio[cyt])" % (scaleo, osm),
    "membrane": "mem",
    "custom_dynamics": True,
    "scale_by_area": False,
}
# # CURRENTS/LEAKS ----------------------------------------------------------------
# sodium (Na) current
mcReactions["na_current"] = {
    "reactant": "nai[cyt]",
    "product": "nao[ecs]",
    "rate_f": f"{gna} * (rxd.v - {ena})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# potassium (K) current
mcReactions["k_current"] = {
    "reactant": "kki[cyt]",
    "product": "kko[ecs]",
    "rate_f": f"{gk}* (rxd.v - {ek})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
# nkcc1 (Na+/K+/2Cl- cotransporter)
mcReactions["nkcc1_current1"] = {
    "reactant": "cli[cyt]",
    "product": "clo[ecs]",
    "rate_f": f"2.0 * {nkcc1} * {volume_scale}",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current2"] = {
    "reactant": "kki[cyt]",
    "product": "kko[ecs]",
    "rate_f": f"{nkcc1} * {volume_scale}",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current3"] = {
    "reactant": "nai[cyt]",
    "product": "nao[ecs]",
    "rate_f": f"{nkcc1} * {volume_scale}",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
# ## kcc2 (K+/Cl- cotransporter)
mcReactions["kcc2_current1"] = {
    "reactant": "cli[cyt]",
    "product": "clo[ecs]",
    "rate_f": f"{kcc2} * {volume_scale}",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
mcReactions["kcc2_current2"] = {
    "reactant": "kki[cyt]",
    "product": "kko[ecs]",
    "rate_f": f"{kcc2} * {volume_scale}",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

## sodium leak
mcReactions["na_leak"] = {
    "reactant": "nai[cyt]",
    "product": "nao[ecs]",
    "rate_f": f"gnabar_l * (rxd.v - {ena})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## potassium leak
mcReactions["k_leak"] = {
    "reactant": "kki[cyt]",
    "product": "kko[ecs]",
    "rate_f": f"gkbar_l * (rxd.v - {ek})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
# ## chlorine (Cl) leak
mcReactions["cl_current"] = {
    "reactant": "cli[cyt]",
    "product": "clo[ecs]",
    "rate_f": f"gclbar_l * ({ecl} - rxd.v)",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
# ## Na+/K+ pump current in neuron (2K+ in, 3Na+ out)
mcReactions["pump_current_k"] = {
    "reactant": "2*kko[ecs]",
    "product": "2*kki[cyt]",
    "rate_f": pumpRate,
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["pump_current_na"] = {
    "reactant": "3*nai[cyt]",
    "product": "3*nao[ecs]",
    "rate_f": pumpRate,
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
mcReactions["pump_current_others"] = {
    "reactant": "ATP[cyt]",
    "product": "ADP[cyt] + Pos[cyt]",
    "rate_f": pumpRate,
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": False,
}
# 6 o2 -> 30 ATP
# assumed o2 in cyt (mM) == o2 in ecs (mM) due to rapid flux across the membrane (~42cm/s == 420um/ms)
mcReactions["O2Flux"] = {
    "reactant": f"{o2ecs}",
    "product": f"{o2cyt}", 
    "rate_f": f"420e18 *({o2ecs} - {o2cyt})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": False,
}

mcReactions["O2Consumed"] = {
    "reactant": "dump[cyt]",
    "product": "o2_consumed[ecs]", 
    "rate_f": f"420e18 *({o2ecs} - {o2cyt})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": False,
}

netParams.rxdParams["multicompartmentReactions"] = mcReactions

reactions = {}
## Glial O2 depletion
"""
reactions["glia_oxygen"] = {
    "reactant": o2ecs,
    "product": "o2_consumed[ecs]",
    "rate_f": "(1/5) * (%s)" % (gliapump),
    "custom_dynamics": True,
}
"""
reactions["ATPRestore"] = {
    "reactant": f"5*ADP[cyt] + 5*Pos[cyt] + {o2cyt}",
    "product": f"5*ATP[cyt]", 
    "rate_f": Vatp,
    "custom_dynamics": True,
}
reactions["BasalATP"] = {
    "reactant": f"ATP",
    "product": f"ADP[cyt] + Pos[cyt]", 
    "rate_f": VBasalATP,
    "custom_dynamics": True,
}
reactions["adenylateKinase"] = {
    "reactant": "AMP[cyt] + ATP[cyt]",
    "product": "2*ADP[cyt]",
    "rate_f" : f"{AMPswitch} * {ATPswitch} * {fnum}/{fden}",
    "rate_b" : f"{ADPswitch} * {bnum}/{bden}",
    "custom_dynamics": True,
}
netParams.rxdParams["reactions"] = reactions
# RATES--------------------------------------------------------------------------
rates = {}
## dm/dt
rates["m_gate"] = {
    "species": "mgate",
    "regions": ["mem"],
    "rate": f"(({alpha_m}) * (1.0 - mgate)) - (({beta_m}) * mgate)",
}

## dh/dt
rates["h_gate"] = {
    "species": "hgate",
    "regions": ["mem"],
    "rate": f"(({alpha_h}) * (1.0 - hgate)) - (({beta_h}) * hgate)",
}

## dn/dt
rates["n_gate"] = {
    "species": "ngate",
    "regions": ["mem"],
    "rate": f"(({alpha_n}) * (1.0 - ngate)) - (({beta_n}) * ngate)",
}

rates["PRestore"] = {
    "species": f"Pos",
    "regions": ["cyt"],
    "rate": f"(1-(Pos/{cfg.Pss}))/{cfg.tauP}",
}

netParams.rxdParams["rates"] = rates

# # plot statistics for 10% of cells
# scale = 10 #max(1,int(sum(N_[:8])/1000))
# include = [(pop, range(0, netParams.popParams[pop]['numCells'], scale)) for pop in L]
# cfg.analysis['plotSpikeStats']['include'] = include


# v0.0 - combination of netParams from ../uniformdensity and netpyne PD thalamocortical model
# v0.1 - got rid of all non cortical cells, background stim
# v0.2 - adding connections back in
# v0.3 - scaled original connections by 1e-7
# v0.4 - original poisson, rather than netstim inputs
# v0.5 - now includes thalamocortical inputs
# v0.6 - different weight scaling for excitation and inhibition
# v0.7 - using netParams.scaleConnWeight (which excludes netstims)
# v0.8 - normally distributed connection weights based on whats worked for fixed conns
# v1.0 - added in o2 sources based on capillaries identified from histology
# v1.1 - updated parameters to avoid SD in baseline simulations
# v1.2 - use pump rather than pump_max to determine leak conductance (initial o2 is lower than max).
