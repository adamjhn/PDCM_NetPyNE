import pandas as pd
import numpy as np
import bluepyopt as bpop
import pickle
import bluepyopt.ephys as ephys
from ipyparallel import Client
from defs import pasInit

import json
from bpoRxD import (
    RxDCellModel,
    RxDSpeciesParameter,
    RxDReactionParameter,
    RxDConstantParameter,
)
from neuron.units import mM, sec, ms

import efel
from datetime import datetime

rc = Client(profile="slurm")
lview = rc.load_balanced_view()
count = 0


def mapper(func, it):
    global count
    start_time = datetime.now()
    print("mapping", start_time, count)
    count += 1
    ret = lview.map_sync(func, it)
    return ret


print("Connected", rc.ids)


class NrnVecStimStimulus(ephys.stimuli.SynapticStimulus):

    """Current stimulus based on current amplitude and time series"""

    def __init__(self, total_duration=None, locations=None, times=None, weights=None):
        """Constructor
        Args:
            location: synapse point process location to connect to
            times: time times of synaptic events
            weights: the corresponding weights for each event
        """

        super(NrnVecStimStimulus, self).__init__()
        self.total_duration = max(times) if total_duration is None else total_duration
        self.locations = locations
        self.times = times
        self.weights = weights
        self.connections = {}

    def instantiate(self, sim=None, icell=None):
        """Run stimulus"""

        if not self.times or self.weights is None:
            return
        if not isinstance(self.times, neuron.hoc.HocObject):
            self.times = sim.neuron.h.Vector(self.times)
        if hasattr(self.weights, "__len__") and not isinstance(
            self.times, neuron.hoc.HocObject
        ):
            self.weights = sim.neuron.h.Vector(self.weights)
        for location in self.locations:
            self.connections[location.name] = []
            for synapse in location.instantiate(sim=sim, icell=icell):
                vecstim = sim.neuron.h.VecStim()
                vecstim.play(self.times)
                netcon = sim.neuron.h.NetCon(vecstim, synapse)
                if hasattr(self.weights, "__len__"):
                    if not isinstance(self.weights, neuron.hoc.HocObject):
                        self.weights = sim.neuron.h.Vector(self.weights)
                    self.weights.play(netcon._ref_weight[0], self.times, 0)
                    netcon.weight[0] = self.weights.x[0]
                else:
                    netcon.weight[0] = self.weights
                self.connections[location.name].append((netcon, vecstim))

    def destroy(self, sim=None):
        """Destroy stimulus"""

        self.connections = {}

    def __str__(self):
        """String representation"""

        return (
            "VecStim at %s" % ",".join(location for location in self.locations)
            if self.locations is not None
            else "VecStim"
        )


# population and example
pop = "L6e"
# load the data
df = pd.read_json("PDMCExample.json")
opt_condtions = json.load(open("OptCondsFixed.json", "r"))

allstims = []
for exidx in range(5):
    stims = {}
    for tsyn, wsyn, msyn, ssyn, lsyn in zip(
        df[pop]["inputs"][exidx],
        df[pop]["weight"][exidx],
        df[pop]["mechs"][exidx],
        df[pop]["secs"][exidx],
        df[pop]["locs"][exidx],
    ):
        label = f"{msyn}_{ssyn}({lsyn})"
        if label in stims:
            if stims[label]["input"][-1] == tsyn:
                print(tsyn, wsyn, stims[label]["weight"][-1])
                stims[label]["weight"][-1] += wsyn
            else:
                stims[label]["input"].append(tsyn)
                stims[label]["weight"].append(wsyn)
        else:
            stims[label] = {"input": [tsyn], "weight": [wsyn]}

    allstims.append(stims)

cells = json.load(open("cells/CSTR_cellParams.json", "r"))


# soma only model
morph = ephys.morphologies.NrnFileMorphology(
    "CellMorph.swc", do_replace_axon=False, nseg_frequency=1e9
)
somatic_loc = ephys.locations.NrnSeclistCompLocation(
    name="soma", seclist_name="somatic", sec_index=0, comp_x=0.5
)

all_loc = ephys.locations.NrnSeclistLocation("all", seclist_name="all")


locs = {"somatic": "soma", "apical": "Adend1", "axonal": "axon", "basal": "Bdend"}
loc_lookup = {}
for loc in locs:
    loc_lookup[loc] = ephys.locations.NrnSeclistLocation(loc, seclist_name=loc)
loc_lookup["all"] = ephys.locations.NrnSeclistLocation("all", seclist_name="all")


bpo_mechs = {}
bpo_params = {}
default_params = {}
for lockey, s in locs.items():
    params = cells["secs"][s]
    loc = loc_lookup[lockey]
    for mech, mparams in params["mechs"].items():
        if mech not in bpo_mechs:
            bpo_mechs[f"{mech}"] = ephys.mechanisms.NrnMODMechanism(
                name=mech,
                suffix=mech,
                locations=[
                    loc_lookup[a]
                    for a, b in locs.items()
                    if mech in cells["secs"][b]["mechs"]
                ],
            )
        for k, v in mparams.items():
            # optimize conductance parameters
            if k.startswith("g"):
                bpo_params[
                    f"{k}_{mech}.{lockey}"
                ] = ephys.parameters.NrnSectionParameter(
                    name=f"{k}_{mech}.{lockey}",
                    param_name=f"{k}_{mech}",
                    locations=[
                        loc
                    ],  # [loc_lookup[a] for a,b in locs.items() if mech in cells['secs'][b]['mechs']],
                    bounds=[0, max(10 * v, 1e-3)],
                    frozen=False,
                )
                default_params[f"{k}_{mech}.{lockey}"] = v
            else:
                bpo_params[
                    f"{k}_{mech}.{lockey}"
                ] = ephys.parameters.NrnSectionParameter(
                    name=f"{k}_{mech}.{lockey}",
                    param_name=f"{k}_{mech}",
                    locations=[
                        loc
                    ],  # [loc_lookup[a] for a,b in locs.items() if mech in cells['secs'][b]['mechs']],
                    value=v,
                    frozen=True,
                )
    for p in ["Ra", "cm"]:
        bpo_params[f"{p}.{lockey}"] = ephys.parameters.NrnSectionParameter(
            name=f"{p}.{lockey}",
            param_name=p,
            locations=[
                loc
            ],  # [loc_lookup[a] for a,b in locs.items() if mech in cells['secs'][b]['mechs']],
            value=params["geom"][p],
            frozen=True,
        )
for ion, ionparams in params["ions"].items():
    if ion == "ca":
        bpo_mechs[f"{ion}_ion"] = ephys.mechanisms.NrnMODMechanism(
            name=f"{ion}_ion", suffix=f"{ion}_ion", locations=[loc_lookup["all"]]
        )
        bpo_params[f"{ion}i0_{ion}_ion"] = ephys.parameters.NrnGlobalParameter(
            name=f"{ion}i0_{ion}_ion",
            param_name=f"{ion}i0_{ion}_ion",
            value=ionparams["i"],
            frozen=True,
        )
        bpo_params[f"{ion}o0_{ion}_ion"] = ephys.parameters.NrnGlobalParameter(
            name=f"{ion}o0_{ion}_ion",
            param_name=f"{ion}o0_{ion}_ion",
            value=ionparams["o"],
            frozen=True,
        )


for k, v in cells["globals"].items():
    bpo_params[f"{k}_h"] = ephys.parameters.NrnGlobalParameter(
        name=k, param_name=k, value=v, frozen=True
    )


# synaptic model
synapses = {
    "exc": {"params": {"tau1": 0.8, "tau2": 5.3, "e": 0}, "loc": somatic_loc},
    "inh": {"params": {"tau1": 0.6, "tau2": 8.5, "e": -75}, "loc": somatic_loc},
}

bposyn = {}
for k, v in synapses.items():
    bposyn[k] = {}
    bpo_mechs[k] = ephys.mechanisms.NrnMODPointProcessMechanism(
        name=k, suffix="Exp2Syn", locations=[v["loc"]]
    )
    bposyn[k]["loc"] = ephys.locations.NrnPointProcessLocation(
        f"{k}_loc", pprocess_mech=bpo_mechs[k]
    )
    for p, val in v["params"].items():
        bpo_params[f"{p}_{k}"] = ephys.parameters.NrnPointProcessParameter(
            name=f"{p}_{k}",
            param_name=p,
            value=val,
            locations=[bposyn[k]["loc"]],
            frozen=True,
        )


### constants
e_charge = 1.60217662e-19
scale = 1e-14 / e_charge
alpha = 5.3
constants = {
    "e_charge": e_charge,
    "scale": scale,
    "gnabar_l": (0.0247 / 1000) * scale,
    "gkbar_l": (0.05 / 1000) * scale,
    "gclbar_l": (0.1 / 1000) * scale,
    "ukcc2": 0.3 * mM / sec,
    "unkcc1": 0.1 * mM / sec,
    "alpha": alpha,
    "epsilon_k_max": 0.25 / sec,
    "epsilon_o2": 0.17 / sec,
    "vtau": 1 / 250.0,
    "g_gliamax": 5 * mM / sec,
    "beta0": 7.0,
    "avo": 6.0221409 * (10**23),
    "p_max": 0.8,  # * mM/sec,
    "nao_initial": 144.0,
    "nai_initial": 18.0,
    "gnai_initial": 18.0,
    "gki_initial": 80.0,
    "ko_initial": 3.5,
    "ki_initial": 140.0,
    "clo_initial": 130.0,
    "cli_initial": 6.0,
    "o2_bath": 0.1,
    "v_initial": -70.0,
    "r0": 100.0,
    "k0": 70.0,
}


### regions
regions = {}

#### ecs dimensions
x = [-10, 60]
y = [-70, 140]
z = [-10, 60]
Vics = 3034.353050318532  # volume of the ICS
Vtotal = 70 * 210 * 70  # total volume
Vscale = 0.8 * Vtotal / Vics  # amount to scale the ICS by so it forms 0.8 of
# total volume i.e. volume_fraction 1/Vscale
regions["ecs"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": 70,
    "volume_fraction": 1 / Vscale,
    "tortuosity": 1.6,
}

regions["ecs_o2"] = {
    "extracellular": True,
    "xlo": x[0],
    "xhi": x[1],
    "ylo": y[0],
    "yhi": y[1],
    "zlo": z[0],
    "zhi": z[1],
    "dx": 70,
    "volume_fraction": 1.0,
    "tortuosity": 1.0,
}

regions["cyt"] = {
    "secs": "all",
    "nrn_region": "i",
    "geometry": {
        "class": "FractionalVolume",
        "args": {"volume_fraction": 0.83, "surface_fraction": 1},
    },
}

regions["mem"] = {"secs": "all", "nrn_region": None, "geometry": "membrane"}


### species
species = {}

k_init_str = "ki_initial if isinstance(node, rxd.node.Node1D) else ko_initial"
species["k"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 2.62,
    "charge": 1,
    "initial": k_init_str,
    "ecs_boundary_conditions": constants["ko_initial"],
    "name": "k",
}

species["na"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 1.78,
    "charge": 1,
    "initial": "nai_initial if isinstance(node, rxd.node.Node1D) else nao_initial",
    "ecs_boundary_conditions": constants["nao_initial"],
    "name": "na",
}

species["cl"] = {
    "regions": ["cyt", "mem", "ecs"],
    "d": 2.1,
    "charge": -1,
    "initial": "cli_initial if isinstance(node, rxd.node.Node1D) else clo_initial",
    "ecs_boundary_conditions": constants["clo_initial"],
    "name": "cl",
}

species["o2_extracellular"] = {
    "regions": ["ecs_o2"],
    "d": 3.3,
    "initial": constants["o2_bath"],
    "ecs_boundary_conditions": constants["o2_bath"],
    "name": "o2_extracellular",
}


### parameters
rxdparams = {}
rxdparams["dump"] = {"regions": ["cyt", "ecs", "ecs_o2"], "name": "dump"}

rxdparams["ecsbc"] = {
    "regions": ["ecs", "ecs_o2"],
    "name": "ecsbc",
    "value": "1 if (abs(node.x3d - ecs._xlo) < ecs._dx[0] or abs(node.x3d - ecs._xhi) < ecs._dx[0] or abs(node.y3d - ecs._ylo) < ecs._dx[1] or abs(node.y3d - ecs._yhi) < ecs._dx[1] or abs(node.z3d - ecs._zlo) < ecs._dx[2] or abs(node.z3d - ecs._zhi) < ecs._dx[2]) else 0",
}

### states
states = {"vol_ratio": {"regions": ["cyt", "ecs"], "initial": 1.0, "name": "vol_ratio"}}

### reactions
fko = "1.0 / (1.0 + rxd.rxdmath.exp(16.0 - k[ecs] / vol_ratio[ecs]))"
nkcc1A = "rxd.rxdmath.log((k[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (k[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1B = "rxd.rxdmath.log((na[cyt] * cl[cyt] / vol_ratio[cyt]**2) / (na[ecs] * cl[ecs] / vol_ratio[ecs]**2))"
nkcc1 = "unkcc1 * (%s) * (%s+%s)" % (fko, nkcc1A, nkcc1B)
kcc2 = "ukcc2 * rxd.rxdmath.log((k[cyt] * cl[cyt] * vol_ratio[cyt]**2) / (k[ecs] * cl[ecs] * vol_ratio[ecs]**2))"

# Nerst equation - reversal potentials
ena = "26.64 * rxd.rxdmath.log(na[ecs]*vol_ratio[cyt]/(na[cyt]*vol_ratio[ecs]))"
ek = "26.64 * rxd.rxdmath.log(k[ecs]*vol_ratio[cyt]/(k[cyt]*vol_ratio[ecs]))"
ecl = "26.64 * rxd.rxdmath.log(cl[cyt]*vol_ratio[ecs]/(cl[ecs]*vol_ratio[cyt]))"

o2ecs = "o2_extracellular[ecs_o2]"
o2switch = "(1.0 + rxd.rxdmath.tanh(1e4 * (%s - 5e-4))) / 2.0" % (o2ecs)
p = "%s * p_max / (1.0 + rxd.rxdmath.exp((20.0 - (%s/vol_ratio[ecs]) * alpha)/3.0))" % (
    o2switch,
    o2ecs,
)
pumpA = "(%s / (1.0 + rxd.rxdmath.exp((25.0 - na[cyt] / vol_ratio[cyt])/3.0)))" % (p)
pumpB = "(1.0 / (1.0 + rxd.rxdmath.exp(3.5 - k[ecs] / vol_ratio[ecs])))"
pump = "(%s) * (%s)" % (pumpA, pumpB)
gliapump = (
    "(1.0/3.0) * (%s / (1.0 + rxd.rxdmath.exp((25.0 - gnai_initial) / 3.0))) * (1.0 / (1.0 + rxd.rxdmath.exp(3.5 - k[ecs]/vol_ratio[ecs])))"
    % (p)
)
g_glia = (
    "g_gliamax / (1.0 + rxd.rxdmath.exp(-((%s)*alpha/vol_ratio[ecs] - 2.5)/0.2))"
    % (o2ecs)
)
glia12 = "(%s) / (1.0 + rxd.rxdmath.exp((18.0 - k[ecs] / vol_ratio[ecs])/2.5))" % (
    g_glia
)

epsilon_kA = (
    "(epsilon_k_max/(1.0 + rxd.rxdmath.exp(-((%s/vol_ratio[ecs]) * alpha - 2.5)/0.2)))"
    % (o2ecs)
)
epsilon_kB = "(1.0/(1.0 + rxd.rxdmath.exp((-20 + ((1.0+1.0/beta0 - vol_ratio[ecs])/vol_ratio[ecs]) /2.0))))"
epsilon_k = "%s * %s" % (epsilon_kA, epsilon_kB)

sa2v = 3.0
volume_scale = "1e-18 * avo * %f" % (1.0 / sa2v)

avo = 6.0221409 * (10**23)
osm = "(1.1029 - 0.1029*rxd.rxdmath.exp( ( (na[ecs] + k[ecs] + cl[ecs] + 18.0)/vol_ratio[ecs] - (na[cyt] + k[cyt] + cl[cyt] + 132.0)/vol_ratio[cyt])/20.0))"
scalei = str(avo * 1e-18)
scaleo = str(avo * 1e-18)

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
# nkcc1 (Na+/K+/2Cl- cotransporter)
mcReactions["nkcc1_current1"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "2.0 * (%s) * (%s)" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current2"] = {
    "reactant": "k[cyt]",
    "product": "k[ecs]",
    "rate_f": "%s * %s" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["nkcc1_current3"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "%s * %s" % (nkcc1, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## kcc2 (K+/Cl- cotransporter)
mcReactions["kcc2_current1"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "%s * %s" % (kcc2, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["kcc2_current2"] = {
    "reactant": "k[cyt]",
    "product": "k[ecs]",
    "rate_f": "%s * %s" % (kcc2, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

## sodium leak
mcReactions["na_leak"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "gnabar_l * (rxd.v - %s)" % (ena),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## potassium leak
mcReactions["k_leak"] = {
    "reactant": "k[cyt]",
    "product": "k[ecs]",
    "rate_f": "gkbar_l * (rxd.v - %s)" % (ek),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## chlorine (Cl) leak
mcReactions["cl_current"] = {
    "reactant": "cl[cyt]",
    "product": "cl[ecs]",
    "rate_f": "gclbar_l * (%s - rxd.v)" % (ecl),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# ## Na+/K+ pump current in neuron (2K+ in, 3Na+ out)
mcReactions["pump_current"] = {
    "reactant": "k[cyt]",
    "product": "k[ecs]",
    "rate_f": "(-2.0 * %s * %s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

mcReactions["pump_current_na"] = {
    "reactant": "na[cyt]",
    "product": "na[ecs]",
    "rate_f": "(3.0 * %s * %s)" % (pump, volume_scale),
    "membrane": "mem",
    "custom_dynamics": True,
    "membrane_flux": True,
}

# RATES--------------------------------------------------------------------------
rates = {}

## diffusion
# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : 'ecsbc * (epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))' % (o2ecs)} # original

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s/vol_ratio[ecs]))' % (o2ecs)} # o2everywhere

# rates['o2diff'] = {'species' : o2ecs, 'regions' : ['ecs_o2'],
#     'rate' : '(epsilon_o2 * (o2_bath - %s))' % (o2ecs)} # o2everywhereNoVolScale

"""
rates["kdiff"] = {
    "species": "k[ecs]",
    "regions": ["ecs"],
    "rate": "ecsbc * ((%s) * (ko_initial - k[ecs]/vol_ratio[ecs]))" % (epsilon_k),
}

rates["nadiff"] = {
    "species": "na[ecs]",
    "regions": ["ecs"],
    "rate": "ecsbc * ((%s) * (nao_initial - na[ecs]/vol_ratio[ecs]))" % (epsilon_k),
}

rates["cldiff"] = {
    "species": "cl[ecs]",
    "regions": ["ecs"],
    "rate": "ecsbc * ((%s) * (clo_initial - cl[ecs]/vol_ratio[ecs]))" % (epsilon_k),
}
"""
## Glia K+/Na+ pump current
rates["glia_k_current"] = {
    "species": "k[ecs]",
    "regions": ["ecs"],
    "rate": "(-(%s) - (2.0 * (%s)))" % (glia12, gliapump),
}

rates["glia_na_current"] = {
    "species": "na[ecs]",
    "regions": ["ecs"],
    "rate": "(3.0 * (%s))" % (gliapump),
}


for p in ["gnabar_l", "gkbar_l", "ukcc2", "unkcc1", "g_gliamax", "p_max"]:
    bpo_params[p] = RxDConstantParameter(
        p, frozen=False, bounds=[0, 10 * constants[p]], param_name=p
    )
    default_params[p] = constants[p]


# cell model
cell = RxDCellModel(
    name=pop,
    morph=morph,
    mechs=list(bpo_mechs.values()),
    params=list(bpo_params.values()),
    regions=regions,
    species=species,
    states=states,
    parameters=rxdparams,
    reactions=None,
    multiCompartmentReactions=mcReactions,
    rates=rates,
    constants=constants,
    initFunc=pasInit,
)


sweep_protocols = []
"""
sweep_protocols = []
for protocol_name, stim in zip([f'cell{i}' for i in range(5)], allstims):
    excstim = NrnVecStimStimulus(locations=[bposyn['exc']['loc']],
                 times=stim['exc_soma(0.5)']['input'],
                 weights=stim['exc_soma(0.5)']['weight'])

    inhstim = NrnVecStimStimulus(locations=[bposyn['inh']['loc']],
                 times=stim['inh_soma(0.5)']['input'],
                 weights=stim['inh_soma(0.5)']['weight'])
    rec_v = ephys.recordings.CompRecording(
            name='%s.soma.v' % protocol_name,
            location=somatic_loc,
            variable='v')
    protocol = ephys.protocols.SweepProtocol(protocol_name, [excstim, inhstim], [rec_v])
    sweep_protocols.append(protocol)
seq_protocol = ephys.protocols.SequenceProtocol('seq', protocols=sweep_protocols)
"""
sweep_protocols = []
for stim in opt_condtions:
    curstim = ephys.stimuli.NrnSquarePulse(
        step_amplitude=float(stim),
        step_delay=100,
        step_duration=500,
        total_duration=1500,
        location=somatic_loc,
    )
    rec_v = ephys.recordings.CompRecording(
        name=f"step{stim}.soma.v", location=somatic_loc, variable="v"
    )
    rec_nai = ephys.recordings.CompRecording(
        name=f"step{stim}.soma.nai", location=somatic_loc, variable="nai"
    )
    rec_ki = ephys.recordings.CompRecording(
        name=f"step{stim}.soma.ki", location=somatic_loc, variable="ki"
    )
    protocol = ephys.protocols.SweepProtocol(
        f"step{stim}", [curstim], [rec_v, rec_nai, rec_ki]
    )
    sweep_protocols.append(protocol)
seq_protocol = ephys.protocols.SequenceProtocol("seq", protocols=sweep_protocols)

nrn = ephys.simulators.NrnSimulator()
# responses = seq_protocol.run(cell_model=cell, param_values=default_params, sim=nrn)

"""
t_vec = np.linspace(0,1000,len(df['L6e']['v_soma'][0]))
features = {}
for pop in df.columns:
    traces = [{'T': t_vec, 'V':  df[pop]['v_soma'][i], 'stim_start':[0], 'stim_end':[1000]} for i in range(5)]
    features[pop] = efel.getFeatureValues(traces, efel.getFeatureNames())

featinc = ["AP1_amp", 'ISI_values', 'AHP1_depth_from_peak', 'Spikecount', 'inv_first_ISI', 'inv_last_ISI']
constraints = {}
for pop in df.columns:
    constraints[pop] = {}
    for f in featinc:
        dat = [features[pop][i][f] for i in range(5) if features[pop][i][f] is not None]
        dat = [i for d in dat for i in d]
        constraints[pop][f] = {'mean': np.nanmean(dat), 'var': np.nanvar(dat)}


objectives = []
for protocol in sweep_protocols:
    feat_list = []
    for feat, vals in constraints['L6e'].items():
        if np.isnan(vals['var']) or vals['var'] == 0: continue
        feat_list.append(ephys.efeatures.eFELFeature(
            feat,
            efel_feature_name=feat,
            stim_start=0,
            stim_end=1000,
            recording_names={'': '%s.soma.v' % protocol.name},
            exp_mean=vals['mean'],
            exp_std=vals['var']**0.5))
    objective = ephys.objectives.WeightedSumObjective(protocol.name,feat_list,[1.0 for _ in range(len(feat_list))])
    objectives.append(objective)
"""

objectives = []
for protocol, opt in zip(sweep_protocols, opt_condtions.values()):
    feat_list = []
    for feat, vals in opt.items():
        feat_list.append(
            ephys.efeatures.eFELFeature(
                feat,
                efel_feature_name=feat,
                stim_start=100,
                stim_end=600,
                recording_names={"": "%s.soma.v" % protocol.name},
                exp_mean=vals["mean"],
                exp_std=vals["std"],
            )
        )
    for ion in ["nai", "ki", "cli"]:
        feat_list.append(
            ephys.efeatures.eFELFeature(
                f"steady_state_{ion}",
                efel_feature_name="steady_state_voltage",
                stim_start=100,
                stim_end=600,
                recording_names={"": "%s.soma.%s" % (protocol.name, ion)},
                exp_mean=constants[f"{ion}_initial"],
                exp_std=0.01 * constants[f"{ion}_initial"],
            )
        )
    objective = ephys.objectives.WeightedSumObjective(
        protocol.name, feat_list, [1.0 for _ in range(len(feat_list))]
    )
    objectives.append(objective)
score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)


cell_evaluator = ephys.evaluators.CellEvaluator(
    cell_model=cell,
    param_names=[p.name for p in bpo_params.values() if not p.frozen],
    fitness_protocols={seq_protocol.name: seq_protocol},
    fitness_calculator=score_calc,
    sim=nrn,
)

optimisation = bpop.optimisations.DEAPOptimisation(
    evaluator=cell_evaluator, map_function=mapper, offspring_size=512
)

final_pop, hall_of_fame, logs, hist = optimisation.run(
    max_ngen=128, continue_cp=True, cp_filename="checkpoints/MCMultiStart.pkl"
)

data = pickle.load(open("checkpoints/MCMulti.pkl", "rb"))
hof = data["halloffame"]
log = data["logbook"]
best_ind_dict = cell_evaluator.param_dict(hof[0])
responses_opt = seq_protocol.run(cell_model=cell, param_values=best_ind_dict, sim=nrn)
print(log)
print(halloffame[0])
print("Done opt")

"""

from neuron import h, rxd
from neuron.units import ms, mV
import numpy as np
import json
import pandas as pd
plt.ion()
h.load_file("stdrun.hoc")
exidx = 1






# set globals
for k,v in cells['globals'].items():
    setattr(h,k,v)

# create the cell model
model = {}
for sc,params in cells['secs'].items():
    sec = h.Section(name=sc)
    model[sc] = sec
    for mech, mparams in params['mechs'].items():
        sec.insert(mech)
        sec.nseg = 1
        for k,v in mparams.items():
            setattr(getattr(sec(0.5),mech), k, v)
    for k,p in params['ions'].items():
        sec.insert(f"{k}_ion")
        setattr(sec, f"{k}i", p['i'])
        setattr(sec, f"{k}o", p['o'])
        setattr(sec, f"e{k}", p['e'])
    for k,v in params['geom'].items():
        if k == 'pt3d':
            sec.pt3dclear()
            for x,y,z,diam3d in v:
                sec.pt3dadd(x,y,z,diam3d)
        else:
            setattr(sec,k,v)
    if params['topol']:
        sec.connect(model[params['topol']['parentSec']], params['topol']['parentX'], params['topol']['childX'])

        syn = h.Exp2Syn(model[ssyn](lsyn))
        syn.tau1 = 0.8 if msyn == 'exc' else 0.6
        syn.tau2 = 5.3 if msyn == 'exc' else 8.5
        syn.e = 0 if msyn == 'exc' else -75
        syns.append(syn)
v_stims = []
v_ncs = []
stim_times = []
for lab, syn in zip (stims,syns):
    vs = h.VecStim()
    stims[lab]['input'] = h.Vector(stims[lab]['input'])
    stims[lab]['weight'] = h.Vector(stims[lab]['weight'])       
    vs.play(stims[lab]['input'])
    nc = h.NetCon(vs, syn)
    nc.weight[0] = stims[lab]['weight'].x[0]
    stims[lab]['weight'].play(nc._ref_weight[0], stims[lab]['input'], 0)
    v_stims.append(vs)
    v_ncs.append(nc)
# setup recording
output_times = h.Vector()
output_times_nc = h.NetCon(model['soma'](0.5)._ref_v, None)
output_times_nc.record(output_times)
t_vec = h.Vector().record(h._ref_t)
somav = h.Vector().record(model['soma'](0.5)._ref_v)
axonv = h.Vector().record(model['axon'](0.5)._ref_v)
# run the simulation
h.dt = 0.01
h.finitialize(h.v_init)

"""
"""
w = nc.weight[0]
print(test_vec.as_numpy()[:5])
print(f"{h.t} weight {w}")
while h.t < 10:
    if nc.weight[0] != w:
        w = nc.weight[0]
        print(f"{h.t} weight {w}")
    h.fadvance()
"""
"""
h.continuerun(1000)


# show a raster plot of the output spikes and the stimulus times
fig, ax = plt.subplots(2,1,figsize=(8, 2))

for c, (color, data) in enumerate([
    ("green", stims['exc_soma(0.5)']['input']),
    ("red", stims['inh_soma(0.5)']['input']),
    ("black", output_times if output_times.size() > 0 else []),
    ("blue", df[pop]['output'][exidx])]):
    ax[0].vlines(data, c - 0.4, c + 0.4, colors=color)

ax[0].set_yticks([0, 1, 2, 3])
ax[0].set_yticklabels(['excitatory\nstimuli', 'inhibitory\nstimuli','output\nevents','expected\noutput'])

ax[0].set_xlim([0, h.t])
ax[0].set_xlabel('time (ms)')

#fig,ax = plt.subplots(2,1,figsize=(8,4))
ax[1].plot(t_vec.as_numpy()[::10], somav.as_numpy()[::10], label='NEURON')
ax[1].plot(t_vec.as_numpy()[::10], df[pop]['v_soma'][exidx], label='NetPyNE')

"""
