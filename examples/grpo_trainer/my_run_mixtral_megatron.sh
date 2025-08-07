#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l select=4
#PBS -l walltime=100:00:00
#PBS -N 0172_verl
#PBS -o outputs/log.txt
#PBS -j oe
#PBS -v RTYPE=rt_HF,USE_SSH=1

set -eux

cd ${PBS_O_WORKDIR}

mkdir -p ${PBS_O_WORKDIR}/outputs/

exec >${PBS_O_WORKDIR}/outputs/${PBS_JOBID}.txt 2>&1

source /etc/profile.d/modules.sh
source ~/.bash_profile
source .venv/bin/activate

module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2

mapfile -t nodes_array < <(awk '{print $1}' $PBS_NODEFILE)
NUM_NODES=${#nodes_array[@]}
GPUS_PER_NODE=8

HF_MODEL_PATH=64expert_2granularity_0shared_top2_52b_active1.4b_z_loss_lr_4e-4_datasets_dclm_dedup_pes2o_math_250b_125b
HF_MODEL_PATH=/groups/gcg51557/experiments/0134_moe_reasoning/checkpoints/megatron/${HF_MODEL_PATH}/iter_0029803
#DIST_CKPT_PATH=megatron-dist/64expert_2granularity_0shared_top2_52b_active1.4b_z_loss_lr_4e-4_datasets_dclm_dedup_pes2o_math_250b_125b
#mkdir -p ${DIST_CKPT_PATH}
DIST_CKPT_PATH=${HF_MODEL_PATH}

head_node=${nodes_array[0]}

head_node_ip=$(ssh ${head_node} "hostname -i" | awk '{print $1}')

PBS_NUMID=$(echo "${PBS_JOBID}" | cut -d '.' -f 1)
port=$((20000 + ($PBS_NUMID % 40000)))
ip_head="${head_node_ip}:${port}"
export ip_head
echo "Head node IP: $ip_head"

echo "Head ノード ${head_node} で Ray ヘッドを起動"
ssh ${head_node} "cd ${PBS_O_WORKDIR} && source /etc/profile.d/modules.sh && source ~/.bash_profile && module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2 && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ray start --head --node-ip-address=${head_node_ip} --port=${port} --dashboard-port=8266 --num-cpus 96 --num-gpus 8 --block" &
sleep 60

worker_nodes=("${nodes_array[@]:1}")
for node in "${worker_nodes[@]}"; do
  echo "ワーカーノード ${node} で Ray ワーカーを起動"
  ssh ${node} "cd ${PBS_O_WORKDIR} && source /etc/profile.d/modules.sh && source ~/.bash_profile && module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2 && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ray start --address ${ip_head} --num-cpus 96 --num-gpus 8 --block" &
done
sleep 60
ssh ${head_node} "cd ${PBS_O_WORKDIR} && source /etc/profile.d/modules.sh && source ~/.bash_profile && module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2 && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ray status"


#python scripts/converter_hf_to_mcore.py --hf_model_path $HF_MODEL_PATH --output_path $DIST_CKPT_PATH

# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1 # For megatron communication/computation overlapping
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WANDB_PRO=verl_grpo_example_gsm8k_math
EXP_NAME=moe_128exp_top2_width2048

ssh ${head_node} "cd ${PBS_O_WORKDIR} && source /etc/profile.d/modules.sh && source ~/.bash_profile && module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2 && source .venv/bin/activate && export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path=$DIST_CKPT_PATH \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PRO} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15"

python scripts/model_merger.py --local_dir checkpoints/${WANDB_PRO}/${EXP_NAME}/global_step_105/actor/ --target_dir checkpoints/${WANDB_PRO}/${EXP_NAME}/global_step_105/actor/huggingface/ --backend megatron
