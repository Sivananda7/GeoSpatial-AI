"""
train_sweeps.py runs hyperparameter search using Weights and Biases and the predefined sweep.yml file
Please adapt the sweep.yml file to your needs and run the script with the following command:
    python scripts/train_sweeps.py --sweep_file sweep.yml

For more information on sweeps in Weights and Biases, please refer to the following link:
https://docs.wandb.ai/guides/sweeps
"""
import wandb
from pathlib import Path
import yaml
import argparse
from train import train, ESDConfig
# import pyprojroot
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


def main():
    wandb.init(project="cs-175-final-project")
    print(wandb.config)
    options = ESDConfig(**wandb.config)
    train(options)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter sweeps using Weights and Biases")
   
    parser.add_argument('--sweep_file', type=str, help="./sweeps.yml", default=None)
    
    parse_args = parser.parse_args()
    
    if parse_args.sweep_file is not None:
        print("not none parse args")
        sweep_file_path = os.path.join(dir_path, parse_args.sweep_file)

        # with open(Path(parse_args.sweep_file)) as f:
        with open(sweep_file_path) as f:

            sweep_config = yaml.safe_load(f)
            print(f"Sweep config: {sweep_config}")

        sweep_id = wandb.sweep(sweep=sweep_config, project="cs-175-final-project")
        wandb.agent(sweep_id, function=main, count=10)