import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
from stats import networkStatsFromOpt
import json
import numpy as np
from cfgSS import cfg
import pandas as pd

# Original PD model stats
target = pd.read_csv('PDNetStats.csv').set_index('population')



# based on single cell trial score < 0.8
gnabar_min, gnabar_max  = 0.01547763659212695, 0.02738686098712858
gkbar_min, gkbar_max    = 0.0072252186353545245, 0.008395290224535777
ukcc2_min, ukcc2_max    = 2.3089414835652383e-05, 0.009995406620321882
unkcc1_min, unkcc1_max  = 1.962743799042841, 2.921144717441156
pmax_min, pmax_max      = 4918.458877405473, 9187.687128667996
gpas_min, gpas_max      = 3.1237344597190856e-05, 6.28561987682117e-05

def batch():
    # parameters space to explore
    params = specs.ODict()
    params["excWeight"] = [0, 0.5]
    params["inhWeightScale"] = [1, 5]
    params["gnabar"] = [gnabar_min, gnabar_max]
    params["gkbar"] = [gkbar_min, gkbar_max]
    params["ukcc2"] = [ukcc2_min, ukcc2_max]
    params["unkcc1"] = [unkcc1_min, unkcc1_max]
    params["pmax"] = [pmax_min, pmax_max]
    params["gpas"] = [gpas_min, gpas_max]

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs["maxFitness"] = 1_000_000_000_000
    fitnessFuncArgs["data"] = (pickle.load(
        open(
            "sample_pd_scale-0.16_DC-0_TH-1_Balanced-1_dur-1.pkl",
            "rb",
        )
    ),
    json.load(open('batchOptNet.json')))
    """
    # 'batchOptNet.json' generated from save network (dat)
    # by pooling cells by type.
    #
    net = {}
    for pop,cell in dat['net']['pops'].items():
        model = cell['tags']['cellModel']
        if model in net:
            net[model]['cellGids'] += cell['cellGids']
        else:
            net[model] = {'cellGids': cell['cellGids']}
    """
    def fitnessFunc(sd, **kwargs):
        print("calc fitness")
        data, net = kwargs["data"]
        # population scores
        stats = networkStatsFromOpt(sd, net, duration=cfg.duration)
        rate_score = 0
        for rt,rs in zip(target['rates'],stats['rates'].values()):
            rate_score += 10*abs(rt-rs)
        
        irregularity_score = 0
        for cvt,cvs in zip(target['irregularity'],stats['irregularity'].values()):
            if np.isnan(cvs):
                irregularity_score += 1e3
            else:
                if cvs>1.0 and cvt<1.0:
                    irregularity_score += 10
                elif cvs<1.0 and cvt>1.0:
                    irregularity_score += 10
                irregularity_score += abs(cvs-cvt)

        synchrony_score = 0
        for st,ss in zip(target['synchrony'],stats['synchrony'].values()):
            diff = abs(st-ss)
            synchrony_score += diff if not np.isnan(diff) else 1e3
        
        # individual cells scores
        spkid = np.array(sd["spkid"])
        spkt = np.array(sd["spkt"]) - 2 # subtract 2ms added delay
        
        score, rate, rxdscore, o2score = 0, 0, 0, 0
        for gid, cell in enumerate(
            [
                f"L{i}{ei}_{idx}"
                for i in [2, 4, 5, 6]
                for ei in ["e", "i"]
                for idx in range(10)
            ]
        ):
            pop, idx = cell.split("_")
            idx = int(idx)
            out = spkt[spkid == gid]
            exp = data[pop]["output"][idx]
            rate += abs(len(exp) - len(out))
            score += dist(out, exp, 2.0)
            for ion in ["k", "na", "cl"]:
                trace = sd[f"{ion}i_soma"][f"cell_{gid}"]
                rxdscore += abs(trace[0] - trace[-1]) / trace[0]
            o2score += sd["o2_consumedo_soma"][f"cell_{gid}"][-1]  # amount of oxygen consumed
        print(f"rate_score {rate_score}\tirregularity_score {irregularity_score}\tsynchrony_score {synchrony_score}")
        print(
            f"rate{rate} score {score}, rxdscore {rxdscore}, o2score {o2score}: {1e3*score + rxdscore + o2score}"
        )
        return min(
            kwargs["maxFitness"],  (rate_score + irregularity_score + synchrony_score) + (rate +  score +  rxdscore) + o2score
        )

    # create Batch object with paramaters to modify, and specifying files to use
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
	'folder': '/ddn/adamjhn/models/PDCM_NetPyNE',
	#'custom': '. "/usr/site/nrniv/local/python/anaconda3/etc/profile.d/conda.sh"\nconda activate py311'
    #'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.batchLabel = "newPumpRate"
    b.saveFolder = "/ddn/adamjhn/data/" + b.batchLabel

    b.optimCfg = {
        "fitnessFunc": fitnessFunc,  # fitness expression (should read simData)
        "fitnessFuncArgs": fitnessFuncArgs,
        "maxFitness": fitnessFuncArgs["maxFitness"],
        "maxiters": 100000,  #    Maximum number of iterations (1 iteration = 1 function evaluation)
        "maxtime": 8 * 60 * 60,  #    Maximum time allowed, in seconds
        "maxiter_wait": 120,
        "time_sleep": 20,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    batch()  # 'simple' or 'complex'
