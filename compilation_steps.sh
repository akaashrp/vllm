if [ -z "${HOSTNAME:-}" ]; then
    host_shortname="$(hostname -s 2>/dev/null)"
    export HOSTNAME="${host_shortname:-unknown-host}"
fi

module load cuda/12.6.1
module load gcc/13.3.1-p20240614

export VLLM_TARGET_DEVICE=cuda
cd $PROJECT/vllm
conda activate vllm
CCACHE_NOHASHDIR="true" VLLM_CPU_DISABLE_AVX512=true \
VLLM_PRECOMPILED_WHEEL_LOCATION=$PROJECT/vllm.whl VLLM_USE_PRECOMPILED=1 \
pip install --no-build-isolation -e . --verbose

VLLM_RUN_SCHEDULER_SIM_TIMING=1 pytest tests/v1/engine/test_scheduler_simulator_native.py -k timing -s
