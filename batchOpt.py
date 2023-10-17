from netpyne import specs
from netpyne.batch import Batch
from scipy import stats
from neuron import h
import numpy as np
import os
from vanRossum import d as dist


def batch():
    # parameters space to explore
    params = specs.ODict()
    params["excWeight"] = [0, 10]
    params["inhWeight"] = [0, 10]

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs["maxFitness"] = 1_000_000_000_000

    def fitnessFunc(sd, **kwargs):
        print("calc fitness")
        spkid = sd["spkid"].as_numpy()
        spkt = sd["spkt"].as_numpy()
        data = kwargs["data"]
        score, rxdscore, o2score = 0, 0, 0
        for cell in sim.net.cells:
            if cell.tags["cellModel"] != "VecStim":
                pop, idx = tags["pop"].split("_")
                dat = data[pop][idx]
                idx = cell.gid
                out = spkt[spkid == cell.gid]
                exp = data[pop]["output"][idx]
                score += dist(out, exp, 2.0)
                soma = cell.secs["soma"]["hObj"]
                for ion in ["k", "na", "cl"]:
                    init = netParams.rxdParams["constants"][f"{ion}i_initial"]
                    val = getattr(soma, f"{ion}i")
                    rxdscore += abs(init - val) / init
                o2score = soma.dumpi  # amount of oxygen consumed
        print(
            f"score {score}, rxdscore {rxdscore}, o2score {o2score}: {1e3*score + rxdscore + o2score}"
        )
        return min(kwargs[maxFitness], 1e3 * score + rxdscore + o2score)

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=params, cfgFile="cfgSS.py", netParamsFile="netParamsSS.py")

    # Set output folder, grid method (all param combinations), and run configuration
    b.method = "optuna"
    b.runCfg = {
        "type": "hpc_slurm",
        "script": "initSS.py",
        # options required only for mpi_direct or hpc
        "mpiCommand": "",
        "nodes": 1,
        "coresPerNode": 1,
        "walltime": "0-00:20:00",
        "partition": "scavenge",
        "allocation": "mcdougal",
        # "email": "adam.newton@yale.edu",
        #'reservation': None,
        "folder": "/home/ajn48/project/PDCM_NetPyNE",
        "custom": """#SBATCH --partition=scavenge
#SBATCH --requeue
#module load miniconda
#module load OpenMPI/4.0.5-GCC-10.2.0 
#source /vast/palmer/apps/avx2/software/miniconda/23.1.0/etc/profile.d/conda.sh
#conda activate py310
"""
        #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.batchLabel = "test"
    print(f"/vast/palmer/scratch/mcdougal/ajn48/{b.batchLabel}")
    b.saveFolder = "/vast/palmer/scratch/mcdougal/ajn48/" + b.batchLabel

    b.optimCfg = {
        "fitnessFunc": fitnessFunc,  # fitness expression (should read simData)
        "fitnessFuncArgs": fitnessFuncArgs,
        "maxFitness": fitnessFuncArgs["maxFitness"],
        "maxiters": 100,  #    Maximum number of iterations (1 iteration = 1 function evaluation)
        "maxtime": 8 * 60 * 60,  #    Maximum time allowed, in seconds
        "maxiter_wait": 60,
        "time_sleep": 10,
    }

    # Run batch simulations
    b.run()


# Main code
if __name__ == "__main__":
    batch()  # 'simple' or 'complex'
