import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
from stats import networkStatsFromOpt
import json
import numpy as np
from cfgPopOpt import cfg
import pandas as pd
from sqlite3 import connect
import os

# Paths
HOMEDIR = "/u/adam"  #'/ddn/adamjhn'
DATADIR = "/tera/adam/data"  #'/ddn/adamjhn/data'
pops = ["L2e"]
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
    # --- Population statistics ---
    stats = networkStatsFromOpt(sd, net, duration=cfg.duration)

    # Rate score: normalized so each population contributes ~0-1
    rate_score = 0
    for pop in pops:
        rt, rs = target["rates"].loc[pop], stats["rates"][pop]
        rate_score += abs(rt - rs) / max(rt, 0.1)
    rate_score /= n_pops  # ~0-1 if rates are within a factor of 2

    # Irregularity score: CV of ISI, target ~0.8-0.9
    irregularity_score = 0
    for pop in pops:
        cvt, cvs = target["irregularity"].loc[pop], stats["irregularity"][pop]
        if np.isnan(cvs):
            irregularity_score += 1.0
        else:
            irregularity_score += min(abs(cvs - cvt), 2.0)
    irregularity_score /= n_pops

    # Synchrony score: Fano factor of population spike counts
    synchrony_score = 0
    for pop in pops:
        st, ss = target["synchrony"].loc[pop], stats["synchrony"][pop]
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
    rxdscore /= n_cells
    atp_score /= n_cells
    o2score /= n_cells

    # Penalty for no spikes: ensures any spiking trial scores better
    no_spike_penalty = 10.0 if len(spkid) == 0 else 0.0
    # Additional penalty of 1/8 for each quite population
    no_spike_pop_penalty = sum([x / len(zeroAP) for x in zeroAP.values()])

    results = {
        "rate_score": rate_score,  # ~0-1: population rates
        "irregularity_score": irregularity_score,  # ~0-1: ISI irregularity
        "synchrony_score": synchrony_score,  # ~0-1: synchrony
        "spike_count_score": spike_count_score,  # ~0-1: per-cell spike count
        "vr_score": vr_score,  # van Rossum distance (unbounded but typically small)
        "rxdscore": 2 * rxdscore,  # ~0-1: ion homeostasis
        "atp_score": atp_score,  # ~0-1: ATP homeostasis
        "o2score": o2score,  # O2 consumption
        "vscore": vscore,  # voltage floor penalty
        "no_spike_penalty": no_spike_penalty,  # penalty for zero spikes
        "no_spike_pop_penalty": no_spike_pop_penalty,
    }

    return results


def batchResults(phase=1):
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
        for pop in pops:
            params[f"excWeight_{pop}"] = [0.001, 0.5]
            params[f"inhWeightScale_{pop}"] = [1, 20]
        label = "phase1_" + "_".join(pops)
    elif phase == 2:
        # Phase 2: narrowed from phase 1 top 20 + 20% margin
        # inhWeightScale lower bound extended to 0.1 to allow inh < exc
        param = {}
        param["excWeight_L2e"] = [0.0151, 0.0305]
        param["excWeight_L4e"] = [0.0189, 0.0615]
        param["excWeight_L5e"] = [0.0046, 0.0415]
        param["excWeight_L6e"] = [0.0001, 1.0]
        param["excWeight_L2i"] = [0.0120, 0.0491]
        param["excWeight_L4i"] = [0.0106, 0.0354]
        param["excWeight_L5i"] = [0.0124, 0.0614]
        param["excWeight_L6i"] = [0.0215, 0.0677]
        param["inhWeightScale_L2e"] = [16.2760, 18.9602]
        param["inhWeightScale_L4e"] = [10.6938, 18.9861]
        param["inhWeightScale_L5e"] = [9.4876, 16.3864]
        param["inhWeightScale_L2i"] = [12.6409, 14.9574]
        param["inhWeightScale_L4i"] = [10.2795, 17.9434]
        param["inhWeightScale_L5i"] = [7.7068, 12.7006]
        param["inhWeightScale_L6i"] = [8.3897, 13.7480]
        param["inhWeightScale_L6e"] = [0, 20]
        for p, v in param.items():
            for pop in pops:
                if pop in p:
                    params[p] = v

        # Allow small biophysical adjustments around single cell optimum
        params["gnabar"] = [0.75 * 0.01385, 1.25 * 0.02500]
        params["gkbar"] = [0.75 * 0.00400, 1.25 * 0.00510]
        params["ukcc2"] = [0.75 * 0.00100, 1.25 * 0.00975]
        params["unkcc1"] = [0.75 * 2.05523, 1.25 * 5.99990]
        params["pmax"] = [0.75 * 5000.01684, 1.25 * 7866.09050]
        params["gpas"] = [0.75 * 0.00003, 1.25 * 0.00005]

        label = "phase2_full_" + "_".join(pops)
    else:
        param = {}
        param["excWeight_L2e"] = [0.0151, 0.0305]
        param["excWeight_L4e"] = [0.0189, 0.0615]
        param["excWeight_L5e"] = [0.0046, 0.0415]
        param["excWeight_L6e"] = [0.0001, 1.0]
        param["excWeight_L2i"] = [0.0120, 0.0491]
        param["excWeight_L4i"] = [0.0106, 0.0354]
        param["excWeight_L5i"] = [0.0124, 0.0614]
        param["excWeight_L6i"] = [0.0215, 0.0677]
        param["inhWeightScale_L2e"] = [16.2760, 18.9602]
        param["inhWeightScale_L4e"] = [10.6938, 18.9861]
        param["inhWeightScale_L5e"] = [9.4876, 16.3864]
        param["inhWeightScale_L2i"] = [12.6409, 14.9574]
        param["inhWeightScale_L4i"] = [10.2795, 17.9434]
        param["inhWeightScale_L5i"] = [7.7068, 12.7006]
        param["inhWeightScale_L6i"] = [8.3897, 13.7480]
        param["inhWeightScale_L6e"] = [0, 20]
        for p, v in param.items():
            for pop in pops:
                if pop in p:
                    params[p] = v

        # Allow small biophysical adjustments around single cell optimum
        params["gnabar"] = [0.75 * 0.01385, 1.25 * 0.02500]
        params["gkbar"] = [0.75 * 0.00400, 1.25 * 0.00510]
        params["ukcc2"] = [0.75 * 0.00100, 1.25 * 0.00975]
        params["unkcc1"] = [0.75 * 2.05523, 1.25 * 5.99990]
        params["pmax"] = [0.75 * 5000.01684, 1.25 * 7866.09050]
        params["gpas"] = [0.75 * 0.00003, 1.25 * 0.00005]

        label = "phase3_full_" + "_".join(pops)

    conn = connect(f"{DATADIR}/{label}/{label}_storage.db")

    query = """SELECT
        trials.number,
        trial_values.trial_id,
        trial_values.objective,
        trial_values.value AS trial_value"""

    for k in params:
        query += f",\nMAX(CASE WHEN trial_params.param_name = '{k}' THEN trial_params.param_value END) AS {k}"

    query += """\nFROM 
        trial_values
    JOIN 
        trial_params ON trial_values.trial_id = trial_params.trial_id,
        trials ON trials.trial_id = trial_values.trial_id
    GROUP BY 
        trial_values.trial_id, trial_values.objective, trial_values.value;
    """

    df = pd.read_sql(query, conn).sort_values("trial_value")
    return df, label


def loadData(number):
    num = number
    try:
        print(f"{DATADIR}/{label}/gen_{num}/trial_{num}_data.json")
        data = json.load(
            open(f"{DATADIR}/{label}/gen_{num}/trial_{num}_data.json", "r")
        )
    except FileNotFoundError:
        return None
    return data


def getScores():
    fitnessFuncArgs = {"maxFitness": 1_000_000_000_000}
    fitnessFuncArgs["data"] = (
        pickle.load(open("sample_pd_scale-0.16_DC-0_TH-1_Balanced-1_dur-1.pkl", "rb")),
        json.load(open("batchOptNet.json")),
    )
    # load results
    results = []
    for k, v in df.iterrows():
        try:
            num = int(v["number"])
            fname = f"{DATADIR}/{label}/gen_{num}/trial_{num}_data.json"
            if os.stat("fname").st_size > 0:
                data = json.load(open(fname, "r"))
            else:
                continue
        except FileNotFoundError:
            continue
        batchres = v.to_dict()
        batchres.update(fitnessFunc(data["simData"], **fitnessFuncArgs))
        results.append(batchres)
    return pd.DataFrame(results)


# Main code
if __name__ == "__main__":
    import sys

    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    df, label = batchResults(phase=phase)
    if len(sys.argv) > 2:
        df = getScores().sort_values("trial_value")

# Check the rxd score of the top 100 results
# main issue is large increase in nai -- is that reflected in rxdscore?
# ignore any that don't spike
df = df[df["no_spike_pop_penalty"] == 0]

A = df[df["trial_value"] < 1.5].sort_values("rxdscore")
