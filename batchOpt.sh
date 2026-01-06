#!/bin/bash
numproc=$1; if [ -z $numproc ]; then numproc=32; fi # Number of processes to use

echo $numproc

for (( i=1; i<=$numproc; i++ ))
    do
        echo "Running batch process $i ..."
        screen -Ldm bash -c "source ~/.bashrc && conda activate py311&& nice python batchOpt.py @"# Run the models
        sleep 20
    done
