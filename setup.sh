pip install -U pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install "sglang[all]==0.4.6.post1" --no-cache-dir --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python
pip install torch-memory-saver --no-cache-dir
pip install --no-cache-dir "vllm==0.8.5.post1" "tensordict==0.6.2" torchdata
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pyext pre-commit ruff
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
pip install -U wheel
pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.4.post1
pip install flashinfer-python==0.2.2.post1+cu124torch2.6 -i https://flashinfer.ai/whl/cu124/torch2.6/
NVTE_FRAMEWORK=pytorch pip3 install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
pip3 install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.0rc3
pip install opencv-python
pip install opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()"
pip install --no-deps -e .
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install sgl-kernel==0.1.1
