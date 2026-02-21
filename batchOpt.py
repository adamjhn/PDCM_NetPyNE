import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
from stats import networkStatsFromOpt
import json
import numpy as np
from cfgSS import cfg
import pandas as pd


# Paths
HOMEDIR='/home/adam' #'/ddn/adamjhn'
DATADIR='/home/adam/models/data' #'/ddn/adamjhn/data'

# Original PD model stats
target = pd.read_csv('PDNetStats.csv').set_index('population')
n_pops = len(target)

# Per-population target rates for normalization
target_rates = target['rates'].values
max_target_rate = max(target_rates.max(), 1.0)


def fitnessFunc(sd, **kwargs):
    """Network fitness with normalized and balanced scoring components."""
    print("calc fitness")
    data, net = kwargs["data"]

    # --- Population statistics ---
    stats = networkStatsFromOpt(sd, net, duration=cfg.duration)

    # Rate score: normalized so each population contributes ~0-1
    rate_score = 0
    for rt, rs in zip(target['rates'], stats['rates'].values()):
        rate_score += abs(rt - rs) / max(rt, 0.1)
    rate_score /= n_pops  # ~0-1 if rates are within a factor of 2

    # Irregularity score: CV of ISI, target ~0.8-0.9
    irregularity_score = 0
    for cvt, cvs in zip(target['irregularity'], stats['irregularity'].values()):
        if np.isnan(cvs):
            irregularity_score += 1.0
        else:
            irregularity_score += min(abs(cvs - cvt), 2.0)
    irregularity_score /= n_pops

    # Synchrony score: Fano factor of population spike counts
    synchrony_score = 0
    for st, ss in zip(target['synchrony'], stats['synchrony'].values()):
        diff = abs(st - ss)
        synchrony_score += min(diff / max(st, 0.1), 2.0) if not np.isnan(diff) else 1.0
    synchrony_score /= n_pops

    # --- Per-cell scores ---
    spkid = np.array(sd["spkid"])
    spkt = np.array(sd["spkt"]) - 2  # subtract 2ms added delay

    cell_list = [
        f"L{i}{ei}_{idx}"
        for i in [2, 4, 5, 6]
        for ei in ["e", "i"]
        for idx in range(10)
    ]
    n_cells = len(cell_list)

    vr_score, spike_count_score = 0, 0
    rxdscore, atp_score, o2score = 0, 0, 0
    vscore = 0
    for gid, cell in enumerate(cell_list):
        pop, idx = cell.split("_")
        idx = int(idx)

        out = spkt[spkid == gid]
        exp = data[pop]["output"][idx]

        # Spike count difference (normalized per cell)
        n_exp = max(len(exp), 1)
        spike_count_score += abs(len(exp) - len(out)) / n_exp

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
    rxdscore /= (n_cells * 3)
    atp_score /= n_cells
    o2score /= n_cells

    # Penalty for no spikes: ensures any spiking trial scores better
    no_spike_penalty = 10.0 if len(spkid) == 0 else 0.0

    total = (
        rate_score              # ~0-1: population rates
        + irregularity_score    # ~0-1: ISI irregularity
        + synchrony_score       # ~0-1: synchrony
        + spike_count_score     # ~0-1: per-cell spike count
        + vr_score              # van Rossum distance (unbounded but typically small)
        + rxdscore              # ~0-1: ion homeostasis
        + atp_score             # ~0-1: ATP homeostasis
        + o2score               # O2 consumption
        + vscore                # voltage floor penalty
        + no_spike_penalty      # penalty for zero spikes
    )

    print(
        f"rate={rate_score:.3f} irr={irregularity_score:.3f} "
        f"sync={synchrony_score:.3f} spkcnt={spike_count_score:.3f} "
        f"vr={vr_score:.3f} rxd={rxdscore:.3f} atp={atp_score:.3f} "
        f"o2={o2score:.3f} v={vscore:.3f} nospk={no_spike_penalty:.0f} "
        f"total={total:.3f}"
    )
    return min(kwargs["maxFitness"], total)


def batch(phase=1):
    """Two-phase network optimization.

    Phase 1: Per-layer synaptic weights (biophysical params fixed from single cell).
    Phase 2: Joint refinement of weights + biophysical params.

    Usage:
        python batchOpt.py          # phase 1 (default)
        python batchOpt.py 1        # phase 1
        python batchOpt.py 2        # phase 2
    """
    params = specs.ODict()

    if phase == 1:
        # Phase 1: per-population synaptic weights
        # Biophysical params fixed at single cell optimum (set in cfgSS.py)
        for pop in cfg.cellPops:
            params[f"excWeight_{pop}"] = [0.001, 0.5]
            params[f"inhWeightScale_{pop}"] = [1, 20]
        label = "phase1_weights"
    else:
        # Phase 2: narrowed from phase 1 top 20 + 20% margin
        # inhWeightScale lower bound extended to 0.1 to allow inh < exc
        params["excWeight_L2e"] = [0.001, 0.577]
        params["excWeight_L2i"] = [0.101, 0.160]
        params["excWeight_L4e"] = [0.171, 0.237]
        params["excWeight_L4i"] = [0.018, 0.088]
        params["excWeight_L5e"] = [0.001, 0.046]
        params["excWeight_L5i"] = [0.123, 0.259]
        params["excWeight_L6e"] = [0.202, 0.341]
        params["excWeight_L6i"] = [0.109, 0.407]
        params["inhWeightScale_L2e"] = [0.1, 17.64]
        params["inhWeightScale_L2i"] = [2.18, 7.64]
        params["inhWeightScale_L4e"] = [0.1, 3.36]
        params["inhWeightScale_L4i"] = [5.32, 7.70]
        params["inhWeightScale_L5e"] = [6.80, 9.01]
        params["inhWeightScale_L5i"] = [0.1, 4.34]
        params["inhWeightScale_L6e"] = [0.1, 3.97]
        params["inhWeightScale_L6i"] = [1.62, 7.47]
        # Allow small biophysical adjustments around single cell optimum
        params["pmax"] = [cfg.pmax*0.9, cfg.pmax*1.1]
        params["gnabar"] = [cfg.gnabar*0.9, cfg.gnabar*1.1]
        label = "phase2_refine"

    fitnessFuncArgs = {"maxFitness": 1_000_000_000_000}
    fitnessFuncArgs["data"] = (
        pickle.load(open("sample_pd_scale-0.16_DC-0_TH-1_Balanced-1_dur-1.pkl", "rb")),
        json.load(open('batchOptNet.json'))
    )

    # create Batch object with parameters to modify, and specifying files to use
    b = Batch(params=params, cfgFile="cfgSS.py", netParamsFile="netParamsSSVecStim.py")

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        'type': 'mpi_direct',
        'script': 'initSSVecStim.py',
        # options required only for hpc
        'mpiCommand': '',
        'executor': '/bin/bash',
        'nodes': 1,
        'coresPerNode': 1,
        'allocation': 'default',
        'email': 'adam.newton@neurosim.downstate.edu',
        'reservation': None,
        'folder': f'{HOMEDIR}/models/PDCM_NetPyNE',
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
        "time_sleep": 20,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    import sys
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    batch(phase=phase)
