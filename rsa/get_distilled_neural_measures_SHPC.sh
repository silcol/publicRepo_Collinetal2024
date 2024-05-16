#!/usr/bin/env bash
#SBATCH -t 200
#SBATCH --mail-user=silvycollin@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name get_distilled_neural_measures_SHPC
#SBATCH -c 1
#SBATCH --constraint=cascade
#SBATCH --array=0-3
#SBATCH --mem=185G


module load anaconda3/2021.5
conda activate wedding-schema
python get_distilled_neural_measures_SHPC.py