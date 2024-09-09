#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="example.py"

# Partition for the job:
#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

# --partition deeplearn
# --qos gpgpudeeplearn

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
#SBATCH --time=00-06:00:0


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

#srun python ours.py  --model Res-GT-GL-cifar10-1.pt --dataset cifar10 --ini_arch 2 2 2 --finale_arch 8 8 8  --LRS Constant  | tee  Res-GT-GL-cifar10-1.txt

fairseq-train /data/gpfs/projects/punim0512/Haihangw-Projects/Neural-Growth-Transformer/data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    #--eval-bleu-print-samples \

#fairseq-generate /data/gpfs/projects/punim0512/Haihangw-Projects/Neural-Growth-Transformer/data-bin/iwslt14.tokenized.de-en \
#    --path /data/gpfs/projects/punim0512/Haihangw-Projects/Neural-Growth-Transformer/checkpoints/checkpoint_best.pt \
#    --batch-size 128 --beam 5 --remove-bpe

