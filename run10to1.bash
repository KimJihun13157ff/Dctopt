#!/bin/bash
#SBATCH --job-name 'DSGD'
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4  
#SBATCH --time 2-0
#SBATCH --partition 3090
#SBATCH --output=pretest_%x_%A.out
#SBATCH --export=ALL,PYTHONUNBUFFERED=1

#module load anaconda3/2022.05


echo "activate dctopt"
source /home/jihunkim/anaconda3/etc/profile.d/conda.sh
conda activate dctopt

echo "Start Training"


###### DSGD , lr [0.01 0.1,0.2]

 #### DCHOCO, same as choco.


 python main.py \
 --gpu_ids 0 --master_port 29403\
 --batch_size 3000 \
 --topology ring  \
 --opt_strategy CHOCO_DZO \
 --num_client 20 \
 --lr 0.01 --max_epoch 500\
 --K 392 --local_iter 10 --gossip_iter 1  --seed 42 --sparsification_r 0.99 --consensus_ss 0.0375


python main.py \
 --gpu_ids 0 --master_port 29403\
 --batch_size 3000 \
 --topology ring  \
 --opt_strategy CHOCO_DZO \
 --num_client 20 \
 --lr 0.1 --max_epoch 500\
 --K 392 --local_iter 10 --gossip_iter 1  --seed 42 --sparsification_r 0.99 --consensus_ss 0.0375

python main.py \
 --gpu_ids 0 --master_port 29403\
 --batch_size 3000 \
 --topology ring  \
 --opt_strategy CHOCO_DZO \
 --num_client 20 \
 --lr 0.2 --max_epoch 500\
 --K 392 --local_iter 10 --gossip_iter 1  --seed 42 --sparsification_r 0.99 --consensus_ss 0.0375



