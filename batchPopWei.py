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
    rxdscore /= n_cells
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
        + rxdscore  # ~0-6: ion homeostasis
        + o2score  # O2 consumption
        + vscore  # voltage floor penalty
        + 10 * no_spike_penalty  # penalty for zero spikes
        + 10 * no_spike_pop_penalty
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
            # param range for top 100 fits to fI curve
            params["gnabar"] = [1.3423e-02, 2.7863e-02]
            params["gkbar"] = [1.4131e-03, 1.0222e-02]
            params["ukcc2"] = [6.0787e-01, 1.2847e00]
            params["unkcc1"] = [1.8316e00, 4.7665e00]
            params["pmax"] = [1.9383e01, 7.0006e01]
            params["gpas"] = [6.4633e-08, 1.3207e-05]

        label = "phase1_wei_" + "_".join(pops)
    elif phase == 2:
        # use the results from phase 1 for weights and modify other
        # params
        for pop in pops:
            if pop == "L2e":
                params["excWeight_L2e"] = [0.75 * 1.0673e-03, 1.25 * 1.1137e-02]
                params["inhWeightScale_L2e"] = [0.75 * 3.0374e00, 1.25 * 6.1293e00]
                params["gnabar"] = [0.75 * 1.3423e-02, 1.25 * 2.1281e-02]
                params["gkbar"] = [0.75 * 1.5339e-03, 1.25 * 1.0219e-02]
                params["ukcc2"] = [0.75 * 6.0942e-01, 1.25 * 7.8374e-01]
                params["unkcc1"] = [0.75 * 2.3719e00, 1.25 * 3.1377e00]
                params["pmax"] = [0.75 * 2.9588e01, 1.25 * 3.9337e01]
                params["gpas"] = [0.75 * 2.1909e-07, 1.25 * 9.4016e-06]
            if pop == "L2i":
                params["excWeight_L2i"] = [0.75 * 1.0098e-03, 1.25 * 1.4528e-03]
                params["inhWeightScale_L2i"] = [0.75 * 3.9150e00, 1.25 * 5.5408e00]
                params["gnabar"] = [0.75 * 1.8295e-02, 1.25 * 2.1831e-02]
                params["gkbar"] = [0.75 * 8.5274e-03, 1.25 * 1.0220e-02]
                params["ukcc2"] = [0.75 * 6.0994e-01, 1.25 * 7.4552e-01]
                params["unkcc1"] = [0.75 * 3.2804e00, 1.25 * 4.0593e00]
                params["pmax"] = [0.75 * 3.3690e01, 1.25 * 6.8453e01]
                params["gpas"] = [0.75 * 8.8509e-06, 1.25 * 1.2478e-05]
            if pop == "L4e":
                params["excWeight_L4e"] = [0.75 * 6.7591e-03, 1.25 * 9.5212e-03]
                params["inhWeightScale_L4e"] = [0.75 * 5.4192e00, 1.25 * 6.6952e00]
                params["gnabar"] = [0.75 * 1.3424e-02, 1.25 * 1.8392e-02]
                params["gkbar"] = [0.75 * 5.6268e-03, 1.25 * 8.4744e-03]
                params["ukcc2"] = [0.75 * 6.1047e-01, 1.25 * 8.9941e-01]
                params["unkcc1"] = [0.75 * 2.3048e00, 1.25 * 4.6647e00]
                params["pmax"] = [0.75 * 2.1820e01, 1.25 * 6.9988e01]
                params["gpas"] = [0.75 * 7.3670e-06, 1.25 * 1.2654e-05]
            if pop == "L4i":
                params["excWeight_L4i"] = [0.75 * 7.0138e-03, 1.25 * 2.3541e-02]
                params["inhWeightScale_L4i"] = [0.75 * 1.1347e00, 1.25 * 9.8699e00]
                params["gnabar"] = [0.75 * 1.3423e-02, 1.25 * 2.7571e-02]
                params["gkbar"] = [0.75 * 5.7413e-03, 1.25 * 1.0222e-02]
                params["ukcc2"] = [0.75 * 6.9146e-01, 1.25 * 1.2504e00]
                params["unkcc1"] = [0.75 * 1.8342e00, 1.25 * 4.7660e00]
                params["pmax"] = [0.75 * 2.2373e01, 1.25 * 7.0001e01]
                params["gpas"] = [0.75 * 1.3742e-06, 1.25 * 1.2803e-05]

            if pop == "L5e":
                params["excWeight_L5e"] = [0.75 * 1.0012e-03, 1.25 * 1.3136e-03]
                params["inhWeightScale_L5e"] = [0.75 * 3.3898e00, 1.25 * 4.9652e00]
                params["gnabar"] = [0.75 * 1.5833e-02, 1.25 * 2.0922e-02]
                params["gkbar"] = [0.75 * 2.4968e-03, 1.25 * 1.0208e-02]
                params["ukcc2"] = [0.75 * 8.8253e-01, 1.25 * 1.1366e00]
                params["unkcc1"] = [0.75 * 2.0251e00, 1.25 * 4.5767e00]
                params["pmax"] = [0.75 * 4.7010e01, 1.25 * 6.8377e01]
                params["gpas"] = [0.75 * 8.1904e-07, 1.25 * 5.8539e-06]
            if pop == "L5i":
                params["excWeight_L5i"] = [0.75 * 6.5518e-03, 1.25 * 9.5801e-03]
                params["inhWeightScale_L5i"] = [0.75 * 4.9211e00, 1.25 * 7.8452e00]
                params["gnabar"] = [0.75 * 1.4798e-02, 1.25 * 2.2359e-02]
                params["gkbar"] = [0.75 * 1.4619e-03, 1.25 * 9.8231e-03]
                params["ukcc2"] = [0.75 * 6.6555e-01, 1.25 * 1.0005e00]
                params["unkcc1"] = [0.75 * 1.9656e00, 1.25 * 4.6334e00]
                params["pmax"] = [0.75 * 2.3602e01, 1.25 * 6.9890e01]
                params["gpas"] = [0.75 * 6.4816e-07, 1.25 * 9.3097e-06]

            if pop == "L6e":
                params["excWeight_L6e"] = [0.75 * 1.0337e-03, 1.25 * 8.3645e-03]
                params["inhWeightScale_L6e"] = [0.75 * 1.2261e00, 1.25 * 3.3772e00]
                params["gnabar"] = [0.75 * 1.8999e-02, 1.25 * 2.0584e-02]
                params["gkbar"] = [0.75 * 4.7810e-03, 1.25 * 6.5728e-03]
                params["ukcc2"] = [0.75 * 7.1454e-01, 1.25 * 8.4827e-01]
                params["unkcc1"] = [0.75 * 1.8318e00, 1.25 * 2.0100e00]
                params["pmax"] = [0.75 * 4.1655e01, 1.25 * 6.8343e01]
                params["gpas"] = [0.75 * 8.1327e-06, 1.25 * 1.1284e-05]
            if pop == "L6i":
                params["excWeight_L6i"] = [0.75 * 1.0022e-03, 1.25 * 1.6081e-03]
                params["inhWeightScale_L6i"] = [0.75 * 1.8119e00, 1.25 * 3.3564e00]
                params["gnabar"] = [0.75 * 2.2387e-02, 1.25 * 2.6949e-02]
                params["gkbar"] = [0.75 * 1.9407e-03, 1.25 * 1.0221e-02]
                params["ukcc2"] = [0.75 * 6.1126e-01, 1.25 * 7.7179e-01]
                params["unkcc1"] = [0.75 * 3.5529e00, 1.25 * 4.5937e00]
                params["pmax"] = [0.75 * 1.9384e01, 1.25 * 6.8010e01]
                params["gpas"] = [0.75 * 2.4965e-06, 1.25 * 1.1731e-05]
        label = "phase2_wei_" + "_".join(pops)
    else:
        # based on top 100 results
        # allow larger pmax, gkbar -- smaller execWeight
        for pop in pops:
            if pop == "L6e":
                params["gnabar"] = [1.3676e-02, 1.5464e-02]
                params["gkbar"] = [5.4516e-03, 5 * 6.3101e-03]  # increase
                params["ukcc2"] = [4.2957e-03, 9.9022e-03]
                params["unkcc1"] = [3.0947e00, 7.4250e00]
                params["pmax"] = [8.8202e03, 5 * 9.8324e03]  # increase
                params["gpas"] = [2.9563e-05, 5.2775e-05]
                params["excWeight_L6e"] = [
                    4.0425e-02 / 10,
                    1.1 * 4.0512e-02,
                ]  # decrease
                params["inhWeightScale_L6e"] = [3.8108e00 / 10, 1.1 * 4.2558e00]
            if pop == "L2e":
                params["excWeight_L2e"] = [0.1 * 3.9526e-02, 1.1 * 4.0078e-02]
                params["inhWeightScale_L2e"] = [0.1 * 6.0518e00, 1.1 * 6.5843e00]
                params["gnabar"] = [0.9 * 1.3993e-02, 1.1 * 1.5980e-02]
                params["gkbar"] = [0.9 * 3.9408e-03, 1.5 * 6.3016e-03]
                params["ukcc2"] = [0.9 * 4.9623e-03, 1.1 * 1.1035e-02]
                params["unkcc1"] = [0.9 * 1.5464e00, 1.1 * 5.7518e00]
                params["pmax"] = [0.9 * 5.3192e03, 1.1 * 8.9720e03]
                params["gpas"] = [0.9 * 4.3354e-05, 1.1 * 5.9247e-05]
        label = "phase3_full_" + "_".join(pops)
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
        netParamsFile="netParamsPopWei.py",
    )

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        "type": "mpi_direct",
        "script": "initPopWei.py",
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
        "maxiters": 500_000_000,
        "maxtime": 5 * 24 * 60 * 60,
        "maxiter_wait": 280,
        "time_sleep": 10,
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
