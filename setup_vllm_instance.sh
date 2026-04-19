#!/bin/bash
# =============================================================================
# vLLM + Gemma 4 Instance Setup Script
# =============================================================================
# Usage (from local machine):
#   ssh -i <key.pem> ubuntu@<host> 'bash -s' < setup_vllm_instance.sh
#
# Tested on: p5en.48xlarge, Ubuntu DLAMI, 8x H200
# Takes ~10 minutes (mostly model download)
# =============================================================================

set -euo pipefail

echo "============================================"
echo "vLLM + Gemma 4 Setup"
echo "Started: $(date -u)"
echo "============================================"

export DATA=/opt/dlami/nvme
mkdir -p $DATA/models $DATA/vllm-env

# --- Install python3-venv if needed ---
echo "[1/4] Checking python3-venv..."
if ! python3 -m venv --help &>/dev/null 2>&1; then
    echo "  Installing python3-venv..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3.10-venv
fi

# --- Create venv and install vLLM ---
echo "[2/4] Installing vLLM..."
if [ ! -f $DATA/vllm-env/bin/activate ]; then
    python3 -m venv $DATA/vllm-env
fi
source $DATA/vllm-env/bin/activate
pip install --upgrade pip -q
pip install vllm fastapi uvicorn httpx -q
echo "vLLM $(python -c 'import vllm; print(vllm.__version__)')"

# --- Copy responses proxy ---
echo "[3/4] Setting up responses proxy..."
cat > $DATA/responses_proxy_launcher.sh << 'PROXY_SCRIPT'
#!/bin/bash
source /opt/dlami/nvme/vllm-env/bin/activate
cd /opt/dlami/nvme
VLLM_BASE_URL=http://localhost:8002/v1 python responses_proxy.py --port 9010 2>&1 | tee responses-proxy.log
PROXY_SCRIPT
chmod +x $DATA/responses_proxy_launcher.sh

# --- Launch vLLM endpoints ---
echo "[4/4] Launching vLLM endpoints..."

# Gemma 4 256K context (GPUs 0-3, TP=4)
screen -dmS gemma-256k bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve google/gemma-4-31B-it \
  --dtype float16 \
  --max-model-len 262144 \
  --tensor-parallel-size 4 \
  --port 8001 \
  --host 0.0.0.0 \
  --download-dir $DATA/models \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  2>&1 | tee $DATA/gemma-256k.log
"

# Gemma 4 32K context (GPU 4)
screen -dmS gemma-32k bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=4
vllm serve google/gemma-4-31B-it \
  --dtype float16 \
  --max-model-len 32768 \
  --port 8002 \
  --host 0.0.0.0 \
  --download-dir $DATA/models \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  2>&1 | tee $DATA/gemma-32k.log
"

# GLM-4.7-Flash (GPU 5)
screen -dmS glm-flash bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=5
vllm serve zai-org/GLM-4.7-Flash \
  --dtype float16 \
  --max-model-len 32768 \
  --port 8003 \
  --host 0.0.0.0 \
  --download-dir $DATA/models \
  --trust-remote-code \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 \
  2>&1 | tee $DATA/glm-flash.log
"

echo ""
echo "============================================"
echo "SETUP COMPLETE"
echo "============================================"
echo "Endpoints (will be ready in ~5-10 min for model download):"
echo "  Port 8001: Gemma 4 31B (256K context, TP=4, GPUs 0-3)"
echo "  Port 8002: Gemma 4 31B (32K context, GPU 4)"
echo "  Port 8003: GLM-4.7-Flash (32K context, GPU 5)"
echo "  Port 9010: Responses API proxy (start manually with responses_proxy.py)"
echo ""
echo "SSH tunnel: ssh -i <key> -L 8002:localhost:8002 -L 9010:localhost:9010 ubuntu@<host>"
echo ""
echo "Monitor: screen -ls"
echo "Check: for p in 8001 8002 8003; do curl -s localhost:\$p/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])'; done"
echo ""
echo "Finished: $(date -u)"
