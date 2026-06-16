#!/usr/bin/env bash
#TODO:
#1. Bad port after initializing instant
#2. Cleaning up bad instances is always unsuccessful.
set -euo pipefail

# ─── Flags ────────────────────────────────────────────────────────────────────
AUTO=false
for arg in "$@"; do
  case $arg in
    --auto) AUTO=true ;;
  esac
done

# ─── Step 0: Preflight checks ─────────────────────────────────────────────────
if ! command -v vastai &>/dev/null; then
  echo "ERROR: vastai CLI not found."
  echo "Install it with: pip install vastai"
  echo "Then set your API key: vastai set api-key <YOUR_KEY>"
  exit 1
fi

echo "Checking Vast.ai authentication..."
if ! vastai show user --raw &>/dev/null 2>&1; then
  echo "ERROR: Vast.ai authentication failed."
  echo "Run: vastai set api-key <YOUR_KEY>"
  echo "Get your key at: https://cloud.vast.ai/account/"
  exit 1
fi
echo "Authenticated."
echo ""

# Read API key (used later in remote_run.sh for server-side self-destruct)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [[ -f "$ENV_FILE" ]]; then
  VASTAI_API_KEY=$(grep -E '^VAST_API_KEY=' "$ENV_FILE" | cut -d '=' -f2-)
fi

if [[ -z "${VASTAI_API_KEY:-}" ]]; then
  echo "ERROR: VAST_API_KEY not found in $ENV_FILE"
  echo "Add a line: VAST_API_KEY=<your_key>"
  exit 1
fi

# Export so all vastai CLI calls in this session pick it up
export VAST_API_KEY="$VASTAI_API_KEY"

# ─── Step 1: Interactive prompts ──────────────────────────────────────────────

# GPU type
GPU_NAMES=("RTX 4090" "RTX 3090" "RTX 3060" "A100 SXM4 80GB" "H100 NVL" "A6000")
GPU_FILTERS=("RTX_4090" "RTX_3090" "RTX_3060" "A100_SXM4_80GB" "H100_NVL" "A6000")

echo "Select GPU type:"
PS3="GPU choice: "
select GPU_LABEL in "${GPU_NAMES[@]}"; do
  if [[ -n "$GPU_LABEL" ]]; then
    IDX=$(( REPLY - 1 ))
    GPU_FILTER="${GPU_FILTERS[$IDX]}"
    echo "  → $GPU_LABEL ($GPU_FILTER)"
    echo ""
    break
  fi
  echo "Invalid selection. Try again."
done

# CUDA version — both images are PyTorch 2.4.x (Python 3.11, satisfies pyproject.toml >=3.11)
CUDA_LABELS=("CUDA 12.1" "CUDA 12.4")
CUDA_MINS=("12.1" "12.4")
DOCKER_IMAGES=(
  "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel"
  "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
)

echo "Select CUDA version:"
PS3="CUDA choice: "
select CUDA_LABEL in "${CUDA_LABELS[@]}"; do
  if [[ -n "$CUDA_LABEL" ]]; then
    IDX=$(( REPLY - 1 ))
    CUDA_MIN="${CUDA_MINS[$IDX]}"
    DOCKER_IMAGE="${DOCKER_IMAGES[$IDX]}"
    echo "  → $CUDA_LABEL (image: $DOCKER_IMAGE)"
    echo ""
    break
  fi
  echo "Invalid selection. Try again."
done

# Entrypoint
ENTRY_LABELS=("main.py — full training run" "main.py test=true — evaluation only")
ENTRY_CMDS=("uv run python main.py" "uv run python main.py test=true")

echo "Select entrypoint:"
PS3="Entrypoint choice: "
select ENTRY_LABEL in "${ENTRY_LABELS[@]}"; do
  if [[ -n "$ENTRY_LABEL" ]]; then
    IDX=$(( REPLY - 1 ))
    ENTRYPOINT_CMD="${ENTRY_CMDS[$IDX]}"
    echo "  → $ENTRY_LABEL"
    echo ""
    break
  fi
  echo "Invalid selection. Try again."
done

# Extra Hydra overrides
read -r -p "Extra Hydra overrides (optional, e.g. env=HalfCheetah-v5 seed=42): " EXTRA_OVERRIDES
echo ""

# Max price per hour
while true; do
  read -r -p "Max price per hour in USD (e.g. 0.50): " MAX_PRICE
  if [[ "$MAX_PRICE" =~ ^[0-9]+(\.[0-9]+)?$ ]] && (( $(echo "$MAX_PRICE > 0" | bc -l) )); then
    echo "  → \$$MAX_PRICE/hr max"
    echo ""
    break
  fi
  echo "Please enter a positive number (e.g. 0.50)."
done

# ─── Step 2: Search for offers ────────────────────────────────────────────────
echo "Searching for available offers..."

# Write JSON to temp file — avoids embedding raw JSON in Python strings (quotes/backslashes break inline parsing)
OFFERS_FILE=$(mktemp /tmp/vastai_offers_XXXXXX)
vastai search offers \
  "gpu_name=${GPU_FILTER} cuda_vers>=${CUDA_MIN} dph<=${MAX_PRICE} rentable=true" \
  -o 'dph' --raw > "$OFFERS_FILE" 2>/dev/null

OFFER_COUNT=$(python3 - "$OFFERS_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
offers = data.get('offers', data) if isinstance(data, dict) else data
print(len(offers))
PYEOF
)

if [[ "$OFFER_COUNT" == "0" ]]; then
  rm -f "$OFFERS_FILE"
  echo "No offers found matching: GPU=$GPU_FILTER, CUDA>=$CUDA_MIN, price<=\$$MAX_PRICE/hr"
  echo "Try relaxing your constraints (higher price, different GPU, or different CUDA version)."
  exit 1
fi

echo ""
echo "Top offers (sorted by price):"
python3 - "$OFFERS_FILE" <<'PYEOF'
import json, sys

def fmt_gb(mb):
    return f"{mb/1024:.0f}GB" if mb else '-'

def fmt_mbps(mbps):
    if not mbps:
        return '-'
    return f"{mbps/1000:.1f}Gbps" if mbps >= 1000 else f"{mbps:.0f}Mbps"

def fmt_cost(c):
    return f"${c:.4f}/GB" if c is not None else '-'

def fmt_rel(r):
    return f"{r*100:.1f}%" if r is not None else '-'

data   = json.load(open(sys.argv[1]))
offers = data.get('offers', data) if isinstance(data, dict) else data

SEP = "─" * 72

for i, o in enumerate(offers[:10], 1):
    price   = o.get('dph_total', o.get('dph_base', 0))
    gpu     = o.get('gpu_name', 'unknown')
    n_gpus  = o.get('num_gpus', 1)
    gpu_str = f"{n_gpus}× {gpu}" if n_gpus and n_gpus > 1 else gpu

    vram    = fmt_gb(o.get('gpu_ram'))
    ram     = fmt_gb(o.get('cpu_ram'))
    disk    = f"{o.get('disk_space', 0):.0f}GB" if o.get('disk_space') else '-'
    cpus    = o.get('cpu_cores_effective', o.get('cpu_cores', '-'))

    loc     = o.get('geolocation') or o.get('country') or '-'
    rel     = fmt_rel(o.get('reliability2'))
    driver  = o.get('driver_version', '-')
    cuda    = o.get('cuda_max_good', o.get('cuda_vers', '-'))

    up_spd  = fmt_mbps(o.get('inet_up'))
    dn_spd  = fmt_mbps(o.get('inet_down'))
    up_cost = fmt_cost(o.get('inet_up_cost'))
    dn_cost = fmt_cost(o.get('inet_down_cost'))
    stor_c  = fmt_cost(o.get('storage_cost'))

    dlperf  = f"{o.get('dlperf', 0):.1f}" if o.get('dlperf') else '-'
    perf_d  = f"{o.get('dlperf_per_dphtotal', 0):.1f}" if o.get('dlperf_per_dphtotal') else '-'
    pcie    = f"{o.get('pcie_bw', 0):.1f}GB/s" if o.get('pcie_bw') else '-'
    static_ip = "yes" if o.get('static_ip') else "no"

    print(SEP)
    print(f"  #{i}  ID: {o['id']}   {gpu_str}   ${price:.3f}/hr")
    print(f"  Hardware  : VRAM {vram}  |  RAM {ram}  |  Disk {disk}  |  CPU {cpus} cores  |  PCIe {pcie}")
    print(f"  Location  : {loc}  |  Reliability {rel}  |  Static IP: {static_ip}")
    print(f"  Internet  : ↑{up_spd} / ↓{dn_spd}  |  Upload cost {up_cost}  |  Download cost {dn_cost}")
    print(f"  Storage   : {stor_c}/mo  |  Driver {driver}  |  CUDA max {cuda}")
    print(f"  DL Perf   : {dlperf} TFLOPS  |  Perf/$ {perf_d}")

print(SEP)
PYEOF
echo ""

# Read offer IDs into array (bash 3.2 compatible — no mapfile)
OFFER_IDS=()
while IFS= read -r line; do
  OFFER_IDS+=("$line")
done < <(python3 - "$OFFERS_FILE" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
offers = data.get('offers', data) if isinstance(data, dict) else data
for o in offers[:10]:
    print(o['id'])
PYEOF
)

if [[ "$AUTO" == "true" ]]; then
  OFFER_ID="${OFFER_IDS[0]}"
  OFFER_PRICE=$(python3 - "$OFFERS_FILE" "0" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
offers = data.get('offers', data) if isinstance(data, dict) else data
o = offers[int(sys.argv[2])]
print(f"{o.get('dph_total', o.get('dph_base', 0)):.3f}")
PYEOF
)
  echo "Auto-selecting cheapest offer: #1 (ID: $OFFER_ID, \$$OFFER_PRICE/hr)"
else
  while true; do
    read -r -p "Enter the number of the offer to use [1-${#OFFER_IDS[@]}]: " SELECTION
    if [[ "$SELECTION" =~ ^[0-9]+$ ]] && (( SELECTION >= 1 && SELECTION <= ${#OFFER_IDS[@]} )); then
      SEL_IDX=$(( SELECTION - 1 ))
      OFFER_ID="${OFFER_IDS[$SEL_IDX]}"
      OFFER_PRICE=$(python3 - "$OFFERS_FILE" "$SEL_IDX" <<'PYEOF'
import json, sys
data = json.load(open(sys.argv[1]))
offers = data.get('offers', data) if isinstance(data, dict) else data
o = offers[int(sys.argv[2])]
print(f"{o.get('dph_total', o.get('dph_base', 0)):.3f}")
PYEOF
)
      echo "  → Selected offer #$SELECTION (ID: $OFFER_ID, \$$OFFER_PRICE/hr)"
      break
    fi
    echo "Please enter a number between 1 and ${#OFFER_IDS[@]}."
  done
fi
rm -f "$OFFERS_FILE"
echo ""

# ─── Confirmation before spending money ───────────────────────────────────────
if [[ "$AUTO" != "true" ]]; then
  read -r -p "Create instance (offer $OFFER_ID) for \$$OFFER_PRICE/hr? [y/N]: " CONFIRM
  if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
  fi
  echo ""
fi

# ─── Step 3: Create instance ──────────────────────────────────────────────────
echo "Creating instance from offer $OFFER_ID..."
INSTANCE_FILE=$(mktemp /tmp/vastai_instance_XXXXXX)
vastai create instance "$OFFER_ID" \
  --image "$DOCKER_IMAGE" \
  --disk 50 \
  --ssh --raw > "$INSTANCE_FILE"

INSTANCE_ID=$(python3 - "$INSTANCE_FILE" <<'PYEOF'
import json, sys
raw = open(sys.argv[1]).read().strip()
if not raw:
    print("ERROR: empty response from vastai", file=sys.stderr)
    sys.exit(1)
d = json.loads(raw)
if d.get('error'):
    print(f"ERROR: {d.get('msg', 'unknown error')} (status {d.get('status_code','')})", file=sys.stderr)
    sys.exit(1)
print(d.get('new_contract', d.get('id', '')))
PYEOF
)
rm -f "$INSTANCE_FILE"

if [[ -z "$INSTANCE_ID" ]]; then
  echo "ERROR: Failed to create instance (no instance ID returned)."
  exit 1
fi

echo "Instance $INSTANCE_ID created."
echo ""

# Local cleanup trap: fires if this script crashes before nohup is launched
_INSTANCE_STARTED=false
_cleanup_instance() {
  [[ "$_INSTANCE_STARTED" == "true" ]] && return 0
  [[ -z "${INSTANCE_ID:-}" ]] && return 0
  echo "Cleaning up instance $INSTANCE_ID..."
  if vastai destroy instance "$INSTANCE_ID" --yes; then
    echo "Instance $INSTANCE_ID destroyed."
  else
    echo "vastai CLI destroy failed, retrying via API..."
    if curl -sf -X DELETE "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
        -H "Authorization: Bearer ${VASTAI_API_KEY}"; then
      echo "Instance $INSTANCE_ID destroyed via API."
    else
      echo "WARNING: Failed to destroy instance $INSTANCE_ID — please remove it manually at https://cloud.vast.ai/"
    fi
  fi
}
trap '_cleanup_instance' EXIT

# ─── Step 4: Wait for instance to be running ──────────────────────────────────
echo "Waiting for instance to start (timeout: 5 min)..."
TIMEOUT=300
ELAPSED=0
while true; do
  # Use vastai show instances (plural) for a reliable JSON structure
  STATUS=$(vastai show instances --raw 2>/dev/null \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
instances = data.get('instances', data) if isinstance(data, dict) else data
match = next((i for i in instances if str(i.get('id','')) == '${INSTANCE_ID}'), {})
print(match.get('actual_status', ''))
" 2>/dev/null || echo "")

  if [[ "$STATUS" == "running" ]]; then
    echo "  Instance is running."
    echo ""
    break
  fi

  if (( ELAPSED >= TIMEOUT )); then
    echo "ERROR: Instance did not start within 5 minutes (status: ${STATUS:-unknown})."
    exit 1
  fi

  echo "  Status: ${STATUS:-unknown} (${ELAPSED}s elapsed)..."
  sleep 10
  ELAPSED=$(( ELAPSED + 10 ))
done

# ─── Step 4b: Wait for SSH to be reachable ────────────────────────────────────
# vastai ssh-url returns either "ssh://user@host:PORT" or "ssh user@host -p PORT"
SSH_FULL=$(vastai ssh-url "$INSTANCE_ID")
SSH_STRIPPED="${SSH_FULL#ssh://}"
if [[ "$SSH_STRIPPED" != "$SSH_FULL" ]]; then
  # URL format: user@host:PORT
  REMOTE_HOST="${SSH_STRIPPED%:*}"
  REMOTE_PORT="${SSH_STRIPPED##*:}"
else
  # "-p PORT" format: strip leading "ssh " then parse
  SSH_ARGS="${SSH_FULL#ssh }"
  REMOTE_HOST=$(echo "$SSH_ARGS" | awk '{print $1}')
  REMOTE_PORT=$(echo "$SSH_ARGS" | awk '{print $NF}')
fi

echo "Waiting for SSH to become reachable on port $REMOTE_PORT..."
SSH_TIMEOUT=120
SSH_ELAPSED=0
until ssh -p "$REMOTE_PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
    -o BatchMode=yes "$REMOTE_HOST" true 2>/dev/null; do
  if (( SSH_ELAPSED >= SSH_TIMEOUT )); then
    echo "ERROR: SSH did not become reachable within ${SSH_TIMEOUT}s."
    exit 1
  fi
  echo "  SSH not ready yet (${SSH_ELAPSED}s elapsed)..."
  sleep 5
  SSH_ELAPSED=$(( SSH_ELAPSED + 5 ))
done
echo "  SSH is ready."
echo ""

# ─── Step 5: Upload code ──────────────────────────────────────────────────────
echo "Uploading project files to instance..."
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use rsync directly — vastai copy does not support --exclude flags
rsync -az --progress \
  -e "ssh -p ${REMOTE_PORT} -o StrictHostKeyChecking=no" \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='outputs/' \
  --exclude='results/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.python-version' \
  "$PROJECT_DIR/" \
  "$REMOTE_HOST:/workspace/"
echo "Upload complete."
echo ""

# ─── Step 6: Install system libs + Python deps ────────────────────────────────
echo "Installing system libraries and Python dependencies..."
# MuJoCo / dm-control require OpenGL/EGL libs not present in the PyTorch Docker image
ssh -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" bash <<'SETUP'
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  libgl1 libosmesa6 libglib2.0-0 libgles2 libegl1
cd /workspace
pip install -q uv
uv sync
SETUP
echo "Dependencies installed."
echo ""

# ─── Step 7: Generate and upload remote_run.sh ────────────────────────────────
TMP_RUNNER=$(mktemp /tmp/remote_run_XXXXXX)
# Note: <<RUNNER (unquoted) intentionally expands variables to bake in INSTANCE_ID and VASTAI_API_KEY
cat > "$TMP_RUNNER" <<RUNNER
#!/usr/bin/env bash
cd /workspace
export MUJOCO_GL=osmesa
echo "[remote] Starting: ${ENTRYPOINT_CMD} ${EXTRA_OVERRIDES}"
${ENTRYPOINT_CMD} ${EXTRA_OVERRIDES}
EXIT_CODE=\$?
echo "[remote] Run finished (exit \$EXIT_CODE). Destroying instance..."
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/${INSTANCE_ID}/" \
  -H "Authorization: Bearer ${VASTAI_API_KEY}" > /dev/null
echo "[remote] Instance destroy request sent."
exit \$EXIT_CODE
RUNNER

scp -P "$REMOTE_PORT" -o StrictHostKeyChecking=no \
  "$TMP_RUNNER" "${REMOTE_HOST}:/workspace/remote_run.sh"
rm -f "$TMP_RUNNER"
echo "Remote runner uploaded."
echo ""

# ─── Step 8: Launch training detached (nohup) ─────────────────────────────────
echo "Launching training (nohup)..."
ssh -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
  "chmod +x /workspace/remote_run.sh && nohup bash /workspace/remote_run.sh > /workspace/training.log 2>&1 &"
echo "Training running on instance $INSTANCE_ID."
echo "Instance will self-destruct when training completes."
echo ""

# Disarm local cleanup trap — instance now owns its own lifecycle
_INSTANCE_STARTED=true

# ─── Step 9: Stream logs ──────────────────────────────────────────────────────
echo "Streaming live logs (Ctrl-C to detach — training continues server-side):"
echo "════════════════════════════════════════════════════════════════════════"
# Wait for log file to appear before tailing (avoids "file not found" race)
ssh -p "$REMOTE_PORT" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
  "until [ -f /workspace/training.log ]; do sleep 1; done; tail -f /workspace/training.log" || true
echo ""
echo "Detached from log stream."
echo "Instance $INSTANCE_ID is still running and will self-destruct when done."
# Use cached SSH_ARGS — don't re-call vastai ssh-url (instance may be gone already)
echo "To reconnect:  ssh -p $REMOTE_PORT -o StrictHostKeyChecking=no $REMOTE_HOST"
