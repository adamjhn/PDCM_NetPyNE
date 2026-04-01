import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
from stats import networkStatsFromOpt
import json
import numpy as np
from cfgPopOpt import cfg
import pandas as pd

# Paths
HOMEDIR = "/u/adam"  #'/ddn/adamjhn'
DATADIR = "/tera/adam/data"  #'/ddn/adamjhn/data

# Original PD model stats
target = pd.read_csv("PDNetStats.csv").set_index("population")
n_pops = len(target)

# Per-population target rates for normalization
target_rates = target["rates"].values
max_target_rate = max(target_rates.max(), 1.0)


def fitnessFunc(sd, **kwargs):
    """Network fitness with normalized and balanced scoring components."""
    print("calc fitness")
    data, net = kwargs["data"]
    pops = kwargs["pops"]
    # --- Population statistics ---
    stats = networkStatsFromOpt(sd, net, duration=cfg.duration)

    # Rate score: normalized so each population contributes ~0-1
    rate_score = 0
    for pop, rs in stats["rates"].items():
        rt = target.loc[pop]["rates"]
        rate_score += abs(rt - rs) / max(rt, 0.1)
    rate_score /= n_pops  # ~0-1 if rates are within a factor of 2

    # Irregularity score: CV of ISI, target ~0.8-0.9
    irregularity_score = 0
    for pop, cvs in stats["irregularity"].items():
        cvt = target.loc[pop]["irregularity"]
        if np.isnan(cvs):
            irregularity_score += 1.0
        else:
            irregularity_score += min(abs(cvs - cvt), 2.0)
    irregularity_score /= n_pops

    # Synchrony score: Fano factor of population spike counts
    synchrony_score = 0
    for pop, ss in stats["synchrony"].items():
        st = target.loc[pop]["synchrony"]
        diff = abs(st - ss)
        synchrony_score += min(diff / max(st, 0.1), 2.0) if not np.isnan(diff) else 1.0
    synchrony_score /= n_pops

    # --- Per-cell scores ---
    spkid = np.array(sd["spkid"])
    spkt = np.array(sd["spkt"]) - 2  # subtract 2ms added delay

    cell_list = [f"{pop}_{idx}" for pop in pops for idx in range(10)]
    n_cells = len(cell_list)

    vr_score, spike_count_score = 0, 0
    rxdscore, atp_score, o2score = 0, 0, 0
    vscore = 0
    zeroAP = {}
    for gid, cell in enumerate(cell_list):
        pop, idx = cell.split("_")
        if pop not in zeroAP:
            zeroAP[pop] = True
        idx = int(idx)

        out = spkt[spkid == gid]
        exp = data[pop]["output"][idx]

        # Spike count difference (normalized per cell)
        n_exp = max(len(exp), 1)
        spike_count_score += abs(len(exp) - len(out)) / n_exp
        if len(out) > 0:
            zeroAP[pop] = False

        # Van Rossum distance
        vr_score += dist(out, exp, 2.0)

        # Ion homeostasis
        for ion in ["k", "na", "cl"]:
            trace = sd[f"{ion}i_soma"][f"cell_{gid}"]
            if trace[0] != 0:
                rxdscore += abs(trace[0] - trace[-1]) / abs(trace[0])

        # ATP homeostasis
        atp = sd["ATPi_soma"][f"cell_{gid}"]
        if atp[0] > 0.01:
            atp_score += abs(atp[0] - atp[-1]) / atp[0]

        # O2 consumption
        o2score += sd["o2_consumedo_soma"][f"cell_{gid}"][-1]

        # Voltage floor
        v_trace = sd["v_soma"][f"cell_{gid}"]
        if len(v_trace) > 0:
            v_min = min(v_trace)
            if v_min < -90:
                vscore += -(v_min + 90) / 10

    # Normalize per-cell scores
    spike_count_score /= n_cells
    vr_score /= n_cells
    rxdscore /= n_cells * 3
    atp_score /= n_cells
    o2score /= n_cells

    # Penalty for no spikes: ensures any spiking trial scores better
    no_spike_penalty = 10.0 if len(spkid) == 0 else 0.0
    # Additional penalty of 1/8 for each quite population
    no_spike_pop_penalty = sum([x / len(zeroAP) for x in zeroAP.values()])

    total = (
        rate_score  # ~0-1: population rates
        + irregularity_score  # ~0-1: ISI irregularity
        + synchrony_score  # ~0-1: synchrony
        + spike_count_score  # ~0-1: per-cell spike count
        + vr_score  # van Rossum distance (unbounded but typically small)
        + rxdscore  # ~0-1: ion homeostasis
        + atp_score  # ~0-1: ATP homeostasis
        + o2score  # O2 consumption
        + vscore  # voltage floor penalty
        + no_spike_penalty  # penalty for zero spikes
        + no_spike_pop_penalty
    )

    print(
        f"rate={rate_score:.3f} irr={irregularity_score:.3f} "
        f"sync={synchrony_score:.3f} spkcnt={spike_count_score:.3f} "
        f"vr={vr_score:.3f} rxd={rxdscore:.3f} atp={atp_score:.3f} "
        f"o2={o2score:.3f} v={vscore:.3f} nospk={no_spike_penalty:.0f} "
        f"total={total:.3f}"
    )
    return min(kwargs["maxFitness"], total)


def batch(phase=1, pops=None):
    """Two-phase network optimization.

    Phase 1: Per-layer synaptic weights (biophysical params fixed from single cell).
    Phase 2: Joint refinement of weights + biophysical params.

    Usage:
        python batchOpt.py          # phase 1 (default)
        python batchOpt.py 1        # phase 1
        python batchOpt.py 2        # phase 2
    """
    if pops is not None:
        cfg.popOpt = pops
        cfg.recordCellsSpikes = [
            f"{pop}_{idx}" for pop in cfg.popOpt for idx in range(10)
        ]
        cfg.recordCells = [f"{pop}_{idx}" for pop in cfg.popOpt for idx in range(10)]
    else:
        pops = cfg.popOpt
    params = specs.ODict()
    if phase == 1:
        # Phase 1: per-population synaptic weights
        # Biophysical params fixed at single cell optimum (set in cfgSS.py)
        for pop in pops:
            params[f"excWeight_{pop}"] = [0.001, 0.5]
            params[f"inhWeightScale_{pop}"] = [1, 20]
        label = "phase1_" + "_".join(pops)
    elif phase == 2:
        # use the results from phase 1 for weights and modify other
        # params
        for pop in pops:
            params["gnabar"] = [0.75 * 0.01385, 1.25 * 0.02500]
            params["gkbar"] = [0.75 * 0.00400, 1.25 * 0.00510]
            params["ukcc2"] = [0.75 * 0.00100, 1.25 * 0.00975]
            params["unkcc1"] = [0.75 * 2.05523, 1.25 * 5.99990]
            params["pmax"] = [0.75 * 5000.01684, 1.25 * 7866.09050]
            params["gpas"] = [0.75 * 0.00003, 1.25 * 0.00005]
            if pop == "L2i":
                params["excWeight_L2i"] = [0.75 * 0.00805, 1.25 * 0.01283]
                params["inhWeightScale_L2i"] = [0.75 * 8.45524, 1.25 * 11.01467]
            if pop == "L4e":
                params["excWeight_L4e"] = [0.75 * 0.00100, 1.25 * 0.00877]
                params["inhWeightScale_L4e"] = [0.75 * 2.09595, 1.25 * 6.27167]
            if pop == "L4i":
                params["excWeight_L4i"] = [0.75 * 0.00102, 1.25 * 0.00984]
                params["inhWeightScale_L4i"] = [0.75 * 3.80757, 1.25 * 9.89350]
            if pop == "L5e":
                params["excWeight_L5e"] = [0.75 * 0.00108, 1.25 * 0.00187]
                params["inhWeightScale_L5e"] = [0.75 * 3.77523, 1.25 * 5.03398]
            if pop == "L5i":
                params["excWeight_L5i"] = [0.75 * 0.00682, 1.25 * 0.00979]
                params["inhWeightScale_L5i"] = [0.75 * 5.18400, 1.25 * 6.05486]
            if pop == "L6e":
                params["excWeight_L6e"] = [0.75 * 0.05390, 1.25 * 0.06738]
                params["inhWeightScale_L6e"] = [0.75 * 4.80210, 1.25 * 5.68063]
            if pop == "L6i":
                params["excWeight_L6i"] = [0.75 * 0.00677, 1.25 * 0.01005]
                params["inhWeightScale_L6i"] = [0.75 * 3.96751, 1.25 * 4.61209]
        label = "phase2_full_" + "_".join(pops)
    else:
        """
        params[excWeight_L2e] = [0.072, 0.484]
        params[excWeight_L2i] = [0.127, 0.136]
        params[excWeight_L4e] = [0.174, 0.188]
        params[excWeight_L4i] = [0.059, 0.070]
        params[excWeight_L5e] = [0.016, 0.045]
        params[excWeight_L5i] = [0.160, 0.201]
        params[excWeight_L6e] = [0.243, 0.296]
        params[excWeight_L6i] = [0.200, 0.260]
        params[inhWeightScale_L2e] = [0.000, 17.153]
        params[inhWeightScale_L2i] = [2.180, 2.847]
        params[inhWeightScale_L4e] = [0.805, 1.191]
        params[inhWeightScale_L4i] = [5.922, 6.459]
        params[inhWeightScale_L5e] = [6.806, 7.849]
        params[inhWeightScale_L5i] = [2.009, 3.965]
        params[inhWeightScale_L6e] = [1.100, 1.402]
        params[inhWeightScale_L6i] = [2.468, 5.211]
        params[pmax] = [4536.282, 5204.591]
        params[gnabar] = [0.021, 0.023]
        """
        # increase range for under firing pops L6i
        # other based on target rate +/- 10%
        params["excWeight_L2i"] = [0.0160, 0.0380]
        params["inhWeightScale_L2i"] = [13.5997, 14.7885]
        params["excWeight_L4e"] = [0.0379, 0.0422]
        params["inhWeightScale_L4e"] = [10.6943, 11.7066]
        params["excWeight_L4i"] = [0.0114, 0.0243]
        params["inhWeightScale_L4i"] = [10.3171, 14.4559]
        params["excWeight_L5e"] = [0.0108, 0.0311]
        params["inhWeightScale_L5e"] = [9.6984, 14.9389]
        params["excWeight_L5i"] = [0.0201, 0.0533]
        params["inhWeightScale_L5i"] = [7.7160, 11.5293]
        params["excWeight_L6e"] = [0.0337, 0.2369]
        params["inhWeightScale_L6e"] = [0.7447, 9.2492]
        params["excWeight_L6i"] = [0.01, 1.0]
        params["inhWeightScale_L6i"] = [0, 5]
        params["pmax"] = [4537.7262, 5292.8469]
        params["gnabar"] = [0.0217, 0.0262]

        label = "phase3_refine"

    fitnessFuncArgs = {"maxFitness": 1_000_000_000_000}
    net = {}
    for i, pop in enumerate(cfg.popOpt):
        net[pop] = {"cellGids": [i for i in range(10 * i, 10 * (i + 1))]}

    fitnessFuncArgs["data"] = (
        pickle.load(open("sample_pd_scale-0.16_DC-0_TH-1_Balanced-1_dur-1.pkl", "rb")),
        net,
    )
    fitnessFuncArgs["pops"] = cfg.popOpt

    # create Batch object with parameters to modify, and specifying files to use
    b = Batch(
        params=params,
        cfg=cfg,
        netParamsFile="netParamsPopOpt.py",
    )

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        "type": "mpi_direct",
        "script": "initSSVecStim.py",
        # options required only for hpc
        "mpiCommand": "",
        "executor": "/bin/bash",
        "nodes": 1,
        "coresPerNode": 1,
        "allocation": "default",
        "email": "adam.newton@neurosim.downstate.edu",
        "reservation": None,
        "folder": f"{HOMEDIR}/models/PDCM_NetPyNE",
        #'custom': '. "/usr/site/nrniv/local/python/anaconda3/etc/profile.d/conda.sh"\nconda activate py311'
    }
    b.batchLabel = label
    b.saveFolder = f"{DATADIR}/{b.batchLabel}"

    b.optimCfg = {
        "fitnessFunc": fitnessFunc,
        "fitnessFuncArgs": fitnessFuncArgs,
        "maxFitness": fitnessFuncArgs["maxFitness"],
        "maxiters": 100_000,
        "maxtime": 8 * 60 * 60,
        "maxiter_wait": 120,
        "time_sleep": 5,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    import sys

    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if len(sys.argv) > 2:
        pops = [sys.argv[i] for i in range(2, len(sys.argv))]
    else:
        pops = None
    batch(phase=phase, pops=pops)
