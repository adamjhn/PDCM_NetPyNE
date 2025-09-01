"""
NetPyNE version of single cells from the Potjans and Diesmann thalamocortical network with multicompartment neurons

netParams.py -- contains the network parameters (netParams object)

Modified to include concentration of Na, K, Cl and O2 using RxD.

"""

from netpyne import specs
import numpy as np
from neuron.units import sec, mM
import math
import json
import pickle


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


############################################################
#
#                    NETWORK PARAMETERS
#
############################################################

# Single cell model
amps = np.linspace(0.2, 2, 10)
all_cells = [f"L2e{i}" for i, _ in enumerate(amps)]
cfg.recordCells = all_cells
cfg.recordCellsSpikes = all_cells
L = ["L2e"]
############################################################
# NetPyNE Network Parameters (netParams)
############################################################

netParams = (
    specs.NetParams()
)  # object of class NetParams to store the network parameters
netParams.amps = amps
############################################################
# Populations parameters
############################################################

# population locations
# from Schmidt et al 2018, PLoS Comp Bio, Macaque V1
netParams.sizeX = 100  # x-dimension (horizontal length) size in um
netParams.sizeY = 100  # y-dimension (vertical height or cortical depth) size in um
netParams.sizeZ = 100  # z-dimension (horizontal depth) size in um
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
    "mod": "Exp2Syn",
    "tau1": 0.8,
    "tau2": 5.3,
    "e": 0,
}  # NMDA synaptic mechanism
netParams.synMechParams["inh"] = {
    "mod": "Exp2Syn",
    "tau1": 0.6,
    "tau2": 8.5,
    "e": -75,
}  # GABA synaptic mechanism


############################################################
# NetPyNE Network Parameters (netParams)
############################################################

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
for i, amp in enumerate(amps):
    netParams.popParams[f"L2e{i}"] = {
        "cellType": "L2e",
        "numCells": 1,
        "cellModel": "L2e",
        "xRange": [25, 75],
        "yRange": [25, 75],
    }

    netParams.stimSourceParams[f"IClamp{i}"] = {
        "type": "IClamp",
        "del": 0,
        "dur": 1e9,
        "amp": amp,
    }

    netParams.stimTargetParams[f"IClamp{i}->L2e{i}"] = {
        "source": f"IClamp{i}",
        "sec": "soma",
        "loc": 0.5,
        "conds": {"pop": f"L2e{i}"},
    }
############################################################
# RxD params
############################################################

### constants
e_charge = 1.60217662e-19
scale = 1e-14 / e_charge
alpha = 5.3  # g/mol

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
    "p_max": cfg.pmax * mM / sec,
    "nao_initial": 144.0,
    "nai_initial": 18.0,
    "gnai_initial": 18.0,
    "gki_initial": 80.0,
    "ko_initial": 3.5,
    "ki_initial": 140.0,
    "clo_initial": 130.0,
    "cli_initial": 6.0,
    "o2_bath": cfg.o2_bath,
    "o2_init": cfg.o2_init,
    "v_initial": cfg.hParams["v_init"],
}


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

o2ecs = "o2_extracellular[ecs_o2]"
# Wei model has o2 baseline 32mg/L, i.e. varies between 0 and 6.0mM
# Our model has o2 baseline of 0.06mM bath or 0.04mM initial
# to provide a similar sigmoid curve for the range of oxygen considered
# currents were scaled by 32/0.05 = 640 mg/L/mM
rescale_o2 = 32 * 20
o2switch = "(1.0 + rxd.rxdmath.tanh(1e4 * (%s - 5e-4))) / 2.0" % (o2ecs)
p = f"{o2switch} / (1.0 + rxd.rxdmath.exp((20.0 - ({o2ecs}/vol_ratio[ecs]) * {rescale_o2})/3.0))"
# pump relation to intracellular Na+ and extracellular K+
pumpA = f"(1.0 / (1.0 + rxd.rxdmath.exp(({cfg.KNai} - nai[cyt] / vol_ratio[cyt])/3.0)))"
pumpB = f"(1.0 / (1.0 + rxd.rxdmath.exp({cfg.KKo} - kko[ecs] / vol_ratio[ecs])))"
pump_max = f"p_max * {pumpA} * {pumpB}"  # pump rate with unlimited o2
pump = f"{p} * {pump_max}"  # pump rate scaled by available o2

pumpAg = "(1.0 / (1.0 + rxd.rxdmath.exp((25 - gnai_initial)/3.0)))"
pumpBg = f"(1.0 / (1.0 + rxd.rxdmath.exp({cfg.GliaKKo} - kko[ecs] / vol_ratio[ecs])))"

avo = 6.0221409 * (10 ** 23)
volume_scale = 1e-18 * avo / cfg.sa2v
osm = "(1.1029 - 0.1029*rxd.rxdmath.exp( ( (nao[ecs] + kko[ecs] + clo[ecs] + 18.0)/vol_ratio[ecs] - (nai[cyt] + kki[cyt] + cli[cyt] + 132.0)/vol_ratio[cyt])/20.0))"
scalei = str(avo * 1e-18)
scaleo = str(avo * 1e-18)


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
    "o2_extracellular[ecs_o2]": constants["o2_init"],
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


min_pmax = f"p_max * ({nkcc1} + {kcc2} + {gk} * (v_initial - {ek})/({volume_scale}))/(2*{pump_max})"
pmin = initEval(min_pmax)
if constants["p_max"] < pmin:
    print("Pump current is too low to balance K+ currents")
    print(f"p_max set to {pmin}")
    constants["p_max"] = pmin / initEval(p)

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
kbalance = f"({gk} * (v_initial - {ek}) + {volume_scale} * ({nkcc1} + {kcc2}  -2.0 * {pump}))  / ({ek} - v_initial)"
nabalance = f"({gna} * (v_initial - {ena}) + ({nkcc1} + 3.0 * {pump}) * {volume_scale}) / ({ena} - v_initial)"

constants["gclbar_l"] = initEval(clbalance)
constants["gkbar_l"] = cfg.gkleak_scale * initEval(kbalance)
constants["gnabar_l"] = initEval(nabalance)

if constants["gkbar_l"] < 0:
    raise Exception(f"Negative leak gkbar_l: {constants['gkbar_l']}")
netParams.rxdParams["constants"] = constants

### regions
regions = {}

#### ecs dimensions
# margin = cfg.somaR
x = [0, 200]
y = [-200, 0]
z = [0, 200]

regions["ecs"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": 200,
    "volume_fraction": cfg.alpha_ecs,
    "tortuosity": cfg.tort_ecs,
}

regions["ecs_o2"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": 200,
    "volume_fraction": 1.0,
    "tortuosity": 1.0,
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
params["o2_extracellular"] = {
    "regions": ["ecs_o2"],
    "initial": constants["o2_init"],
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

params["dump"] = {"regions": ["ecs", "ecs_o2", "cyt"], "name": "dump"}


netParams.rxdParams["parameters"] = params

### states
netParams.rxdParams["states"] = {
    "vol_ratio": {"regions": ["cyt", "ecs"], "initial": 1.0, "name": "volume"},
    "mgate": {"regions": ["mem"], "initial": m_initial, "name": "mgate"},
    "hgate": {"regions": ["mem"], "initial": h_initial, "name": "hgate"},
    "ngate": {"regions": ["mem"], "initial": n_initial, "name": "ngate"},
    "o2_consumed": {"regions": ["ecs_o2"], "initial": 0, "name": "o2_consumed"},
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
mcReactions["pump_current"] = {
    "reactant": "kki[cyt]",
    "product": "kko[ecs]",
    "rate_f": f"(-2.0 * {pump} * {volume_scale})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["pump_current_na"] = {
    "reactant": "nai[cyt]",
    "product": "nao[ecs]",
    "rate_f": f"(3.0 * {pump} * {volume_scale})",
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}
# O2 depletrion from Na/K pump in neuron
mcReactions["oxygen"] = {
    "reactant": o2ecs,
    "product": "o2_consumed[ecs_o2]",
    "rate_f": "(1/5) * (%s) * (%s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
}

netParams.rxdParams["multicompartmentReactions"] = mcReactions

reactions = {}
## Glial O2 depletion
"""
reactions["glia_oxygen"] = {
    "reactant": o2ecs,
    "product": "o2_consumed[ecs_o2]",
    "rate_f": "(1/5) * (%s)" % (gliapump),
    "custom_dynamics": True,
}
"""
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


netParams.rxdParams["rates"] = rates

# Single cell rxd model to fit fI curve and minimize energy consumption.
