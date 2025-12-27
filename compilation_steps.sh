module load cuda/12.6.1
module load gcc/13.3.1-p20240614

export VLLM_TARGET_DEVICE=cuda
cd vllm
conda activate vllm
VLLM_CPU_DISABLE_AVX512=true VLLM_PRECOMPILED_WHEEL_LOCATION=$PROJECT/vllm.whl VLLM_USE_PRECOMPILED=1 pip install --no-build-isolation -e . --verbose


CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .