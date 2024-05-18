#! /usr/bin/env python3
import subprocess
import sys
from itertools import product

SCRIPT_NAME = "ppo-discrete.py"
EXP_NAME = "grid-search-rest"
base_cmd = f"nice -10 python3 {SCRIPT_NAME} --exp-name {EXP_NAME} --wandb-project-name {EXP_NAME}"
num_runs = 3
ARGS = {
    "num-steps": [64, 128, 256], 
    "max-grad-norm": [0.3, 0.5, 1.0],
    "update-epochs": [1, 3, 5],
    "alpha": [0.0001, 0.001, 0.01],
    "regularization": [None, 'L1', 'L2']
}

if __name__ == "__main__":
    for args in product(*ARGS.values()):    
        for run in range(num_runs):
            cmd = base_cmd.split(' ')
            cmd += ["--seed", str(run)]
            for arg, val in zip(ARGS.keys(), args):
                cmd += [f"--{arg}", str(val)]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"Error running: {cmd}")
                sys.exit(1)