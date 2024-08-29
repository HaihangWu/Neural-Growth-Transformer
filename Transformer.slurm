#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="example.py"

# Partition for the job:
# --partition=feit-gpu-a100
# --qos=feit

#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn

# -p gpu-a100

# Request different amount of resources for different nodes
# --constraint=[dlg1|dlg2|dlg3]
# --constraint=[dlg4|dlg5]
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=01-00:00:0


#source /usr/local/module/spartan_new.sh
#module load fosscuda/2020b
#module load gcc/10.2.0 cuda/11.1.1 openmpi/4.0.5
#module load python/3.8.6
#module load pytorch/1.9.0-python-3.8.6
#module load torchvision/0.10.0-python-3.8.6-pytorch-1.9.0
#module load matplotlib/3.3.3-python-3.8.6

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda deactivate
conda activate Uniformerv2

# The job command(s):
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "r$SLURM_NODEID Launching python script."

#srun python fixed_model.py  --model Res-4-4-cifar100-1.pt --dataset cifar100 --ini_arch 4 4 4 --finale_arch 4 4 4    | tee  Res-4-4-cifar100-1.txt

#srun python forward_disturbance.py --model Res-lambdaR.8-2-8-cifar100-1.pt --dataset cifar100 --ini_arch 2 2 2 --finale_arch 8 8 8  --copy_random_lambda 0.8 | tee  Res-lambdaR.8-2-8-cifar100-1.txt

#srun python growth-order.py --model Res-p8-ours-cifar100-4.pt --dataset cifar100  --ini_arch 2 2 2 --finale_arch 8 8 8  --growth_interval 8  --GrowthLocation Ours | tee  Res-p8-ours-cifar100-4.txt

#srun python ours.py  --model Res-GT-GL-cifar10-1.pt --dataset cifar10 --ini_arch 2 2 2 --finale_arch 8 8 8  --LRS Constant  | tee  Res-GT-GL-cifar10-1.txt
