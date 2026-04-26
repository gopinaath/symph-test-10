#!/bin/bash
# =============================================================================
# Automated setup and validation for p5en.48xlarge vLLM instances
# =============================================================================
# Usage:
#   # Gemma 4 instance:
#   ./setup_and_validate.sh --host <HOST> --role gemma
#
#   # Kimi K2.6 instance:
#   ./setup_and_validate.sh --host <HOST> --role kimi
#
#   # Both at once:
#   ./setup_and_validate.sh --gemma-host <HOST1> --kimi-host <HOST2>
#
# The script is idempotent — safe to re-run. On failure it cleans up and retries.
# =============================================================================

set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/p5-key-hlwd.pem}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=30"
PROXY_SRC="${PROXY_SRC:-$HOME/dev/scratch/karpathy/vllm-test-10/responses_proxy.py}"
SETUP_SCRIPT="${SETUP_SCRIPT:-$(dirname "$0")/setup_vllm_instance.sh}"
MAX_WAIT=900  # 15 minutes max wait for model loading
CHECK_INTERVAL=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARN:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; }

ssh_cmd() {
    local host="$1"; shift
    ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$host" "$@"
}

scp_cmd() {
    local src="$1" host="$2" dst="$3"
    scp -i "$SSH_KEY" $SSH_OPTS "$src" "$SSH_USER@$host:$dst"
}

# ── Check instance reachability ──
check_instance() {
    local host="$1"
    log "Checking $host..."
    if ssh_cmd "$host" "echo OK && uptime" 2>/dev/null; then
        return 0
    else
        err "Cannot reach $host"
        return 1
    fi
}

# ── Install prerequisites ──
install_prereqs() {
    local host="$1"
    log "Installing prerequisites on $host..."
    ssh_cmd "$host" "sudo apt-get update -qq && sudo apt-get install -y -qq python3.10-venv 2>&1 | tail -1" || {
        warn "apt install failed, trying without update..."
        ssh_cmd "$host" "sudo apt-get install -y -qq python3.10-venv 2>&1 | tail -1"
    }
}

# ── Install vLLM ──
install_vllm() {
    local host="$1"
    log "Installing vLLM on $host..."
    ssh_cmd "$host" bash -s << 'REMOTE'
set -e
export DATA=/opt/dlami/nvme
mkdir -p $DATA/models $DATA/vllm-env

if [ ! -f $DATA/vllm-env/bin/activate ]; then
    python3 -m venv $DATA/vllm-env
fi
source $DATA/vllm-env/bin/activate
pip install --upgrade pip -q
pip install vllm fastapi uvicorn httpx -q 2>&1 | tail -3
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')"
REMOTE
}

# ── Copy responses proxy ──
copy_proxy() {
    local host="$1"
    if [ -f "$PROXY_SRC" ]; then
        log "Copying responses proxy to $host..."
        scp_cmd "$PROXY_SRC" "$host" "/opt/dlami/nvme/responses_proxy.py"
    else
        warn "Proxy source not found at $PROXY_SRC — skipping"
    fi
}

# ── Kill existing vLLM screens ──
kill_screens() {
    local host="$1"
    log "Killing existing screen sessions on $host..."
    ssh_cmd "$host" "screen -ls 2>/dev/null | grep -oP '\d+\.\S+' | xargs -r -I{} screen -X -S {} quit 2>/dev/null; echo 'Cleaned'" || true
}

# ── Launch Gemma 4 endpoints ──
launch_gemma() {
    local host="$1"
    log "Launching Gemma 4 endpoints on $host..."
    ssh_cmd "$host" bash -s << 'REMOTE'
export DATA=/opt/dlami/nvme
source $DATA/vllm-env/bin/activate

# Gemma 4 256K (GPUs 0-3, TP=4)
screen -dmS gemma-256k bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve google/gemma-4-31B-it \
  --dtype float16 --max-model-len 262144 --tensor-parallel-size 4 \
  --port 8001 --host 0.0.0.0 --download-dir $DATA/models \
  --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 2>&1 | tee $DATA/gemma-256k.log
"

# Gemma 4 32K (GPU 4)
screen -dmS gemma-32k bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=4
vllm serve google/gemma-4-31B-it \
  --dtype float16 --max-model-len 32768 \
  --port 8002 --host 0.0.0.0 --download-dir $DATA/models \
  --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 2>&1 | tee $DATA/gemma-32k.log
"

# GLM-4.7-Flash (GPU 5)
screen -dmS glm-flash bash -c "
source $DATA/vllm-env/bin/activate
export CUDA_VISIBLE_DEVICES=5
vllm serve zai-org/GLM-4.7-Flash \
  --dtype float16 --max-model-len 32768 \
  --port 8003 --host 0.0.0.0 --download-dir $DATA/models \
  --trust-remote-code --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --gpu-memory-utilization 0.95 2>&1 | tee $DATA/glm-flash.log
"

# Responses proxy -> Gemma 4 32K
screen -dmS responses-proxy bash -c "
source $DATA/vllm-env/bin/activate
cd $DATA
VLLM_BASE_URL=http://localhost:8002/v1 python responses_proxy.py --port 9010 2>&1 | tee responses-proxy.log
"

sleep 2
screen -ls
REMOTE
}

# ── Launch Kimi K2.6 endpoint ──
launch_kimi() {
    local host="$1"
    log "Launching Kimi K2.6 on $host (all 8 GPUs)..."
    ssh_cmd "$host" bash -s << 'REMOTE'
export DATA=/opt/dlami/nvme
source $DATA/vllm-env/bin/activate

screen -dmS kimi-k2.6 bash -c "
source $DATA/vllm-env/bin/activate
vllm serve moonshotai/Kimi-K2.6 \
  --dtype auto --max-model-len 65536 --tensor-parallel-size 8 \
  --port 8001 --host 0.0.0.0 --download-dir $DATA/models \
  --trust-remote-code --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  2>&1 | tee $DATA/kimi-k2.6.log
"

screen -dmS responses-proxy bash -c "
source $DATA/vllm-env/bin/activate
cd $DATA
VLLM_BASE_URL=http://localhost:8001/v1 python responses_proxy.py --port 9010 2>&1 | tee responses-proxy.log
"

sleep 2
screen -ls
REMOTE
}

# ── Wait for endpoints to become ready ──
wait_for_endpoints() {
    local host="$1"
    shift
    local ports=("$@")
    local elapsed=0

    log "Waiting for endpoints on $host: ${ports[*]}..."
    while [ $elapsed -lt $MAX_WAIT ]; do
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))

        local all_ready=true
        local status_line=""
        for port in "${ports[@]}"; do
            local code
            code=$(ssh_cmd "$host" "curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 http://localhost:$port/v1/models" 2>/dev/null || echo "000")
            status_line+="$port=$code "
            if [ "$code" != "200" ]; then
                all_ready=false
            fi
        done

        log "  [$((elapsed))s] $status_line"

        if $all_ready; then
            log "All endpoints ready on $host!"
            return 0
        fi
    done

    err "Timeout waiting for endpoints on $host after ${MAX_WAIT}s"
    return 1
}

# ── Validate with a test query ──
validate_endpoint() {
    local host="$1" port="$2" model="$3"
    log "Validating $model on $host:$port..."

    local response
    response=$(ssh_cmd "$host" "curl -s http://localhost:$port/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python function add(a,b) that returns a+b. Only the function.\"}],
        \"max_tokens\": 100,
        \"temperature\": 0.0
      }'" 2>/dev/null)

    if echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); assert 'def ' in r['choices'][0]['message']['content']; print(f'OK: {r[\"usage\"][\"total_tokens\"]} tokens')" 2>/dev/null; then
        log "  PASS: $model on $host:$port"
        return 0
    else
        err "  FAIL: $model on $host:$port"
        return 1
    fi
}

# ── Full setup for one instance ──
setup_instance() {
    local host="$1" role="$2"
    local attempt=0 max_attempts=2

    while [ $attempt -lt $max_attempts ]; do
        attempt=$((attempt + 1))
        log "=== Setting up $role on $host (attempt $attempt/$max_attempts) ==="

        # Step 1: Prereqs
        install_prereqs "$host" || { err "Prereqs failed"; continue; }

        # Step 2: Install vLLM
        install_vllm "$host" || { err "vLLM install failed"; continue; }

        # Step 3: Copy proxy
        copy_proxy "$host"

        # Step 4: Kill old screens
        kill_screens "$host"

        # Step 5: Launch
        if [ "$role" = "gemma" ]; then
            launch_gemma "$host" || { err "Launch failed"; continue; }
            wait_for_endpoints "$host" 8001 8002 8003 9010 || { err "Endpoints failed"; kill_screens "$host"; continue; }
            validate_endpoint "$host" 8002 "google/gemma-4-31B-it" || { err "Validation failed"; kill_screens "$host"; continue; }
        elif [ "$role" = "kimi" ]; then
            launch_kimi "$host" || { err "Launch failed"; continue; }
            wait_for_endpoints "$host" 8001 9010 || { err "Endpoints failed"; kill_screens "$host"; continue; }
            validate_endpoint "$host" 8001 "moonshotai/Kimi-K2.6" || { err "Validation failed"; kill_screens "$host"; continue; }
        else
            err "Unknown role: $role"
            return 1
        fi

        log "=== $role on $host: SETUP COMPLETE ==="
        return 0
    done

    err "=== $role on $host: SETUP FAILED after $max_attempts attempts ==="
    return 1
}

# ── Tunnel setup helper ──
setup_tunnels() {
    local gemma_host="$1" kimi_host="$2"

    # Kill existing tunnels
    pkill -f "ssh.*-L.*8002.*$gemma_host" 2>/dev/null || true
    pkill -f "ssh.*-L.*8004.*$kimi_host" 2>/dev/null || true
    sleep 1

    if [ -n "$gemma_host" ]; then
        log "Setting up tunnel to Gemma instance ($gemma_host)..."
        ssh -i "$SSH_KEY" $SSH_OPTS -f -N \
            -L 8002:localhost:8002 \
            -L 9010:localhost:9010 \
            "$SSH_USER@$gemma_host" 2>/dev/null
        log "  Gemma tunnel: localhost:8002 (vLLM), localhost:9010 (proxy)"
    fi

    if [ -n "$kimi_host" ]; then
        log "Setting up tunnel to Kimi instance ($kimi_host)..."
        ssh -i "$SSH_KEY" $SSH_OPTS -f -N \
            -L 8004:localhost:8001 \
            -L 9011:localhost:9010 \
            "$SSH_USER@$kimi_host" 2>/dev/null
        log "  Kimi tunnel: localhost:8004 (vLLM), localhost:9011 (proxy)"
    fi
}

# ── Local validation ──
validate_local() {
    local gemma_host="$1" kimi_host="$2"
    local pass=0 fail=0

    if [ -n "$gemma_host" ]; then
        log "Validating Gemma 4 via local tunnel..."
        if curl -s http://localhost:8002/v1/models | python3 -c "import json,sys; print('Gemma:', json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; then
            ((pass++))
        else
            ((fail++)); err "Gemma local tunnel FAILED"
        fi
    fi

    if [ -n "$kimi_host" ]; then
        log "Validating Kimi K2.6 via local tunnel..."
        if curl -s http://localhost:8004/v1/models | python3 -c "import json,sys; print('Kimi:', json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null; then
            ((pass++))
        else
            ((fail++)); err "Kimi local tunnel FAILED"
        fi
    fi

    log "Local validation: $pass passed, $fail failed"
    [ $fail -eq 0 ]
}

# ── Main ──
GEMMA_HOST="" KIMI_HOST="" ROLE="" HOST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gemma-host) GEMMA_HOST="$2"; shift 2 ;;
        --kimi-host)  KIMI_HOST="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --role)       ROLE="$2"; shift 2 ;;
        *) err "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  vLLM Instance Setup & Validation"
echo "  $(date -u)"
echo "============================================"
echo ""

# Single instance mode
if [ -n "$HOST" ] && [ -n "$ROLE" ]; then
    check_instance "$HOST" || exit 1
    setup_instance "$HOST" "$ROLE"
    exit $?
fi

# Dual instance mode
if [ -n "$GEMMA_HOST" ] || [ -n "$KIMI_HOST" ]; then
    FAILED=0

    if [ -n "$GEMMA_HOST" ]; then
        check_instance "$GEMMA_HOST" || { err "Gemma host unreachable"; FAILED=1; }
        if [ $FAILED -eq 0 ]; then
            setup_instance "$GEMMA_HOST" "gemma" || FAILED=1
        fi
    fi

    if [ -n "$KIMI_HOST" ]; then
        check_instance "$KIMI_HOST" || { err "Kimi host unreachable"; FAILED=1; }
        if [ $FAILED -eq 0 ] || [ -n "$GEMMA_HOST" ]; then
            setup_instance "$KIMI_HOST" "kimi" || FAILED=1
        fi
    fi

    # Set up tunnels
    setup_tunnels "${GEMMA_HOST:-}" "${KIMI_HOST:-}"

    # Local validation
    validate_local "${GEMMA_HOST:-}" "${KIMI_HOST:-}" || FAILED=1

    echo ""
    echo "============================================"
    if [ $FAILED -eq 0 ]; then
        echo "  ALL SETUP COMPLETE"
    else
        echo "  SETUP COMPLETED WITH ERRORS"
    fi
    echo "============================================"
    [ -n "$GEMMA_HOST" ] && echo "  Gemma: $GEMMA_HOST (ports 8001-8003, 9010)"
    [ -n "$KIMI_HOST" ]  && echo "  Kimi:  $KIMI_HOST (port 8001, 9010)"
    echo ""
    echo "  Local tunnels:"
    [ -n "$GEMMA_HOST" ] && echo "    Gemma: localhost:8002 (vLLM), localhost:9010 (proxy)"
    [ -n "$KIMI_HOST" ]  && echo "    Kimi:  localhost:8004 (vLLM), localhost:9011 (proxy)"
    echo ""
    echo "  Test:"
    [ -n "$GEMMA_HOST" ] && echo "    curl localhost:8002/v1/models"
    [ -n "$KIMI_HOST" ]  && echo "    curl localhost:8004/v1/models"
    echo ""
    exit $FAILED
fi

echo "Usage:"
echo "  $0 --host <HOST> --role gemma|kimi"
echo "  $0 --gemma-host <HOST1> --kimi-host <HOST2>"
exit 1
