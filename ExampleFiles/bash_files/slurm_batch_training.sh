#! /bin/sh

#SBATCH -t 2-04:30:00
#SBATCH -J training_job
#SBATCH -n 6
#SBATCH --exclusive
#SBATCH --reservation=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --error=/location/to/logging/directory/training_job.%j.err
#SBATCH --output=/location/to/logging/directory/training_job.%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=YOUR EMAIL ADDRESS TO GET NOTIFIED

# Activate your necessary modules here:
source activate tensorflow

# All your python scripts go here:
/location/to/python/script/model_trainer.py -i /location/to/input/data/directory/ -o /location/to/output/directory/  -c /location/to/model/configuration/directory/ -b 100 -e 10 -r 2 -w 100 -H 5 -V 5 -v 60 -t 60 -M

# Monitoring jobs using "squeue -u $USER"
# Cancel jobs using "scancel JOBID"