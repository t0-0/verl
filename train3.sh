#!/bin/bash
#PBS -P gcg51557
#PBS -q R9920251000
#PBS -l select=1
#PBS -l walltime=168:00:00
#PBS -N 0172
#PBS -o outputs/log.txt
#PBS -j oe
#PBS -v RTYPE=rt_HF

set -euxo pipefail

cd ${PBS_O_WORKDIR}

mkdir -p ${PBS_O_WORKDIR}/outputs
exec >${PBS_O_WORKDIR}/outputs/${PBS_JOBID}.txt 2>&1

source /etc/profile.d/modules.sh
source ~/.bash_profile
source .venv/bin/activate

module load cuda-12.4.1 cudnn-9.8.0 hpcx/2.21.2/modulefiles/hpcx nccl-2.26.2

CHECKPOINTS_DIR=HF_TEST
#rm -rf $CHECKPOINTS_DIR

MODEL_PATH=Qwen/Qwen2.5-Math-1.5B

#export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export DUMP_DIR=${CHECKPOINTS_DIR}/dump
mkdir -p ${DUMP_DIR}

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/one_shot_rlvr/dsr_sub.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=${MODEL_PATH} \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 +actor_rollout_ref.actor.get_hook_stats=False \
 actor_rollout_ref.rollout.name=hf \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.rollout.top_p=1.0 \
 actor_rollout_ref.rollout.top_k=50 \
 +actor_rollout_ref.rollout.val_temperature=1.0 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 +actor_rollout_ref.rollout.model_name=${MODEL_PATH} \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl_few_shot' \
 trainer.experiment_name=${MODEL_PATH}/${CHECKPOINTS_DIR} \
 trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=10 \
 trainer.total_training_steps=2000 2>&1 | tee verl_demo.log
