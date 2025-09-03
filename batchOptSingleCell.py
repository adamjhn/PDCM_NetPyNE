import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
import numpy as np
from cfgSS import cfg
from netParamsSingleCell import netParams


taum = 10  # ms
tauref = 2  # ms
Vrest = -65  # mV
Vth = -50  # mV
Cm = 250  # pF
R = taum / (1e-3 * Cm)  # Mohms
Vreset = -65
Vth = -50

# caclulate target freq
amps = netParams.amps
target = [
    1e3 / (taum * np.log(R * a / (Vreset - Vth + R * a)) + tauref)
    if Vreset - Vth + R * a > 0
    else 0
    for a in amps
]
rheobase = (Vth - Vreset)/R


def batch():
    # parameters space to explore
    params = specs.ODict()
    params["gnabar"] = [1e-4, 1e-1]
    params["gkbar"] = [1e-4, 1e-1]
    params["ukcc2"] = [1e-6, 1]
    params["unkcc1"] = [1e-6, 1]
    params["pmax"] = [1e-6, 100]
    params["gpas"] = [0, 1e-2]

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs["maxFitness"] = 1_000_000

    def fitnessFunc(sd, **kwargs):
        print("calc fitness")

        if len(sd["spkid"]) > 0:
            spkid = np.array(sd["spkid"])
            freq = np.array([sum(spkid == i) for i, _ in enumerate(amps)])
            # range 0-1 unless freq> 2 * target[-1]
            freqscore = sum(abs(target - freq))/len(amps)/target[-1]

            # range 0-1
            rheobaseScore = abs(rheobase-amps[spkid.min()])/(amps[-1]-rheobase)
        else:
            freqscore = 1.0
            rheobaseScore = 1.0

        rxdscore, o2score = 0, 0
        for gid, _ in enumerate(amps):
            for ion in ["k", "na", "cl"]:
                trace = sd[f"{ion}i_soma"][f"cell_{gid}"]
                rxdscore += abs(trace[0] - trace[-1]) / trace[0]
            o2score += sd["o2_consumedo_soma"][f"cell_{gid}"][
                -1
            ]  # amount of oxygen consumed
        print(f"freqscore {freqscore}, rxdscore {rxdscore}, o2score {o2score}")
        return min(kwargs["maxFitness"], 1e2 * (freqscore + rheobaseScore) + rxdscore + o2score)

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, cfgFile="cfgSS.py", netParamsFile="netParamsSingleCell.py")

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        "type": "mpi_direct",
        "script": "initSingleCell.py",
        # options required only for hpc
        "mpiCommand": "mpiexec",
        "nodes": 1,
        "coresPerNode": 1,
        "allocation": "default",
        "email": "adam.newton@neurosim.downstate.edu",
        "reservation": None,
        "folder": "/home/adam/models/PDCM_NetPyNE.BPOCells"
        #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.batchLabel = "cellFit2"
    b.saveFolder = "/ddn/adamjhn/data/" + b.batchLabel

    b.optimCfg = {
        "fitnessFunc": fitnessFunc,  # fitness expression (should read simData)
        "fitnessFuncArgs": fitnessFuncArgs,
        "maxFitness": fitnessFuncArgs["maxFitness"],
        "maxiters": 10000,  #    Maximum number of iterations (1 iteration = 1 function evaluation)
        "maxtime": 8 * 60 * 60,  #    Maximum time allowed, in seconds
        "maxiter_wait": 20,
        "time_sleep": 10,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    batch()  # 'simple' or 'complex'
