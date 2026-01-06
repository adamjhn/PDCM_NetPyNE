import pandas as pd
from sqlite3 import connect
import json
import numpy as np

simLabel = "cellFit6" #"weightsRate"
savepath = f"/tera/adam/data/{simLabel}"
conn = connect(f'{savepath}/{simLabel}_storage.db')
maxFitness = 1_000_000


taum = 10  # ms
tauref = 2  # ms
Vrest = -65  # mV
Vth = -50  # mV
Cm = 250  # pF
R = taum / (1e-3 * Cm)  # Mohms
Vreset = -65
Vth = -50

# caclulate target freq
amps = np.linspace(0.07, 0.7, 10) #netParams.amps
target = [
    1e3 / (taum * np.log(R * a / (Vreset - Vth + R * a)) + tauref)
    if Vreset - Vth + R * a > 0
    else 0
    for a in amps
]
rheobase = (Vth - Vreset)/R



def batch_params():
    params = dict()
    params["excWeight"] = [0, 5e-3]
    params["inhWeightScale"] = [0.1, 10]
    params["gnabar"] = [1e-4, 1e-2]
    params["gkbar"] = [1e-4, 1e-2]
    params["ukcc2"] = [1e-6, 1]
    params["unkcc1"] = [1e-6, 1]
    params["pmax"] = [1e-6, 100]
    params["gpas"] = [0, 0.0001]
    return params


params = batch_params()

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


df = pd.read_sql(query, conn)


def fitnessFunc(sd, **kwargs):
    if len(sd["spkid"]) > 0:
        spkid = np.array(sd["spkid"], dtype=int)
        freq = np.array([sum(spkid == i) for i, _ in enumerate(amps)])
        # range 0-1 unless freq> 2 * target[-1]
        freqscore = sum(abs(target - freq))/len(amps)/target[-1]
         # range 0-1
        rheobaseScore = abs(rheobase-amps[spkid.min()])/(amps[-1]-rheobase)
    else:
        freqscore = 1.0
        rheobaseScore = 1.0
    rxdscore, o2score = 0, 0
    vmin = 0
    for gid, _ in enumerate(amps):
        for ion in ["k", "na", "cl"]:
            trace = sd[f"{ion}i_soma"][f"cell_{gid}"]
            rxdscore += abs(trace[0] - trace[-1]) / trace[0]
            o2score += sd["o2_consumedo_soma"][f"cell_{gid}"][
                -1
            ]  # amount of oxygen consumed
        if len(sd['v_soma'][f"cell_{gid}"])>0:
            vmin = min(vmin, min(sd['v_soma'][f"cell_{gid}"]))
    return freqscore, rheobaseScore, rxdscore, o2score, vmin



freqscores, rheobaseScores, rxdscores, o2scores, vmins = [], [], [], [], []
for num in df['number']:
    try:
        print(f"{savepath}/gen_{num}/trial_{num}_data.json")
        data = json.load(open(f"{savepath}/gen_{num}/trial_{num}_data.json","r"))
    except FileNotFoundError:
        freqscores.append(maxFitness)
        rheobaseScores.append(maxFitness)
        rxdscores.append(maxFitness)
        o2scores.append(maxFitness)
        vmins.append(0)
        continue
    sd = data['simData']
    f, r, rx, ox, v = fitnessFunc(sd)
    freqscores.append(f)
    rheobaseScores.append(r)
    rxdscores.append(rx)
    o2scores.append(ox)
    vmins.append(v)
df['freqScore'] = freqscores
df['rheobaseScore'] = rheobaseScores
df['rxdscore'] = rxdscores
df['o2score'] = o2scores
df['vmin'] = vmins

# filter for reasonable (spiking) results
df = df[df['trial_value']<2].sort_values('trial_value')
idx = df['rxdscore'].argmin()

# print cfg (and results)
for k,v in df.iloc[idx].items():
    print(f"cfg.{k} = {v}")

