import pickle
import numpy as np
import json


def dict_to_latex_table(keys, values):
    # Begin LaTeX table
    latex_table = "\\begin{tabular}{|"

    # Add columns for each key in the dictionary
    for _ in keys:
        latex_table += "c|"
    latex_table += "}\n\\hline\n"

    # Add column names
    for key in keys():
        latex_table += f"{key} & "
    latex_table = latex_table[:-2]  # Remove the last '& ' from the last column name
    latex_table += "\\\\\n\\hline\n"

    # Add values as a row
    if hasattr(values[0], "__len__"):
        for v in values:
            for value in v:
                latex_table += (
                    f"{value:.2f} & " if isinstance(value, (float)) else f"{value} & "
                )
                latex_table = latex_table[
                    :-2
                ]  # Remove the last '& ' from the last value
                latex_table += "\\\\\n"
    else:
        for value in values:
            latex_table += (
                f"{value:.2f} & " if isinstance(value, (float)) else f"{value} & "
            )
            latex_table = latex_table[:-2]  # Remove the last '& ' from the last value
            latex_table += "\\\\\n"
    latex_table += "\\hline\n"

    # End LaTeX table
    latex_table += "\\end{tabular}"

    return latex_table


def networkStatsFromData(dat, filename, N=1000):
    spkt = np.array(dat["simData"]["spkt"])
    spkid = np.array(dat["simData"]["spkid"])
    pops = dat["net"]["pops"][pop]["cellGids"]
    duration = dat["simConfig"]["duration"]
    stats = networkStats(spkt, spkid, pops, duration, N)
    json.dump(stats, open(filename, "w"))


def networkStatsFromSim(sim, filename, N=1000):
    pops = {k: {"cellGids": v.cellGids} for k, v in sim.net.pops.items()}
    spkt = sim.simData["spkt"].as_numpy()
    spkid = sim.simData["spkid"].as_numpy()
    duration = sim.cfg.duration
    stats = networkStats(spkt, spkid, pops, duration, N)
    json.dump(stats, open(filename, "w"))


def networkStats(spkt, spkid, pops, duration, Nsample=1000):
    L = list(pops.keys())
    spkmap = {pop: pops[pop]["cellGids"] for pop in L}
    smpmap = {pop: np.random.choice(spkmap[pop], size=Nsample) for pop in L}
    maxid = spkmap[L[-1]][-1]
    spktimes = {pop: spkt[[i in spkmap[pop] for i in spkid]] for pop in L}
    rates = {pop: 1e3 * len(v) / duration / len(spkmap[pop]) for pop, v in spktimes.items()}

    bins = np.array(range(0, int(duration), 3))
    sample_hist = [np.histogram(st, bins=bins)[0] for st in spktimes.values()]

    sync = {}
    for pop, k in zip(L, sample_hist):
        sync[pop] = k.var() / k.mean()

    isis = [np.diff(spkt[spkid == i]) for i in range(maxid + 1)]

    sample_isi = {pop: np.concatenate([isis[i] for i in smpmap[pop]]) for pop in L}
    irr = {}
    for pop, k in sample_isi.items():
        irr[pop] = k.std() / k.mean()
    return {"rates": rates, "synchrony": sync, "irregularity": irr}
