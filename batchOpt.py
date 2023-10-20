import pickle
from netpyne import specs
from netpyne.batch import Batch
from vanRossum import d as dist
import numpy as np
import cfgSS as cfg


def batch():
    # parameters space to explore
    params = specs.ODict()
    params["excWeight"] = [0, 1]
    params["inhWeight"] = [0, 1]
    params["gnabar"] = [1e-4, 1e-2]
    params["gkbar"] = [1e-4, 1e-2]
    params["ukcc2"] = [1e-6, 1]
    params["unkcc1"] = [1e-6, 1]
    params["pmax"] = [1e-6, 100]
    params["gpas"] = [0, 0.0001]

    # fitness function
    fitnessFuncArgs = {}
    fitnessFuncArgs["maxFitness"] = 1_000_000_000_000
    fitnessFuncArgs["data"] = pickle.load(
        open(
            "/home/ajn48/project/PDCM_NetPyNE/sample_pd_scale-1.0_DC-0_TH-1_Balanced-1_dur-1.pkl",
            "rb",
        )
    )

    def fitnessFunc(sd, **kwargs):
        print("calc fitness")
        spkid = np.array(sd["spkid"])
        spkt = np.array(sd["spkt"])
        data = kwargs["data"]
        score, rxdscore, o2score = 0, 0, 0
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
            score += dist(out, exp, 2.0)
            for ion in ["k", "na", "cl"]:
                trace = sd[f"{ion}i_soma"][f"cell_{gid}"]
                rxdscore += abs(trace[0] - trace[-1]) / trace[0]
            o2score = sd["dumpi_soma"][f"cell_{gid}"][-1]  # amount of oxygen consumed
        print(
            f"score {score}, rxdscore {rxdscore}, o2score {o2score}: {1e3*score + rxdscore + o2score}"
        )
        return min(kwargs["maxFitness"], 1e3 * score + rxdscore + o2score)

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
    b.batchLabel = "weightOpt"
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
