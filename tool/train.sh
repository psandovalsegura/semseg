#!/bin/sh
#SBATCH --account=djacobs
#SBATCH --job-name=pspnet-voc
#SBATCH --time=1-12:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:p6000:4
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail         
#SBATCH --mail-user=psando@umd.edu

# Setup environment
export SCRIPT_DIR="/cfarhomes/psando/Documents/semseg"
export WORK_DIR="/scratch0/slurm_${SLURM_JOBID}"

export CONFIG_FILE="voc2012_pspnet50_val.yaml"
export DATASET="voc2012"
export MODEL="pspnet50"


exp_dir=/vulcanscratch/psando/semseg_experiments/exp/${DATASET}/${MODEL}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${DATASET}/${CONFIG_FILE}
# now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${model_dir} ${result_dir}
cp tool/train.sh tool/train.py tool/test.sh tool/test.py tool/validate.py tool/validate.sh ${config} ${exp_dir}

# Setup environment
python3 -m venv ${WORK_DIR}/tmp-env;
source ${WORK_DIR}/tmp-env/bin/activate;
pip3 install --upgrade pip;
pip3 install -r requirements.txt;

# Train and validate
export PYTHONPATH=${PYTHONPATH}:${SCRIPT_DIR}
python ${exp_dir}/train.py --config=${config}
python ${exp_dir}/validate.py --config=${config} 
