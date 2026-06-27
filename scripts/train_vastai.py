#!/usr/bin/env python3
"""Train on Vast.ai — Python rewrite of train_vastai.sh.

Usage:
    uv run --group vastai python scripts/train_vastai.py [--auto] [--keep-alive]
    make train-vast-py [ARGS="--auto --keep-alive"]
"""

import atexit
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import questionary
import typer

# ── Constants ─────────────────────────────────────────────────────────────────

INSTANCE_START_TIMEOUT = 600  # seconds to wait for instance to reach "running" status

GPU_NAMES = ["RTX 4090", "RTX 3090", "RTX 3060", "A100 SXM4 80GB", "H100 NVL", "A6000"]
GPU_FILTERS = ["RTX_4090", "RTX_3090", "RTX_3060", "A100_SXM4_80GB", "H100_NVL", "A6000"]

CUDA_LABELS = ["CUDA 12.1", "CUDA 12.4"]
CUDA_MINS = ["12.1", "12.4"]
DOCKER_IMAGES = [
    "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel",
    "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel",
]

ENTRY_LABELS = [
    "main.py — full training run",
    "main.py test=true — evaluation only",
]
ENTRY_CMDS = [
    "uv run python main.py",
    "uv run python main.py test=true",
]

# ── Cleanup state ─────────────────────────────────────────────────────────────

_instance_id: str | None = None
_instance_started: bool = False
_vastai_api_key: str = ""
_r2_account_id: str = ""
_r2_access_key: str = ""
_r2_secret_key: str = ""


def _cleanup() -> None:
    if _instance_started or not _instance_id:
        return
    typer.echo(f"\nCleaning up instance {_instance_id}...")
    result = subprocess.run(
        ["vastai", "destroy", "instance", _instance_id, "--yes"],
        capture_output=True,
    )
    if result.returncode == 0:
        typer.echo(f"Instance {_instance_id} destroyed.")
        return
    typer.echo("vastai CLI destroy failed, retrying via API...")
    curl = subprocess.run(
        [
            "curl", "-sf", "-X", "DELETE",
            f"https://console.vast.ai/api/v0/instances/{_instance_id}/",
            "-H", f"Authorization: Bearer {_vastai_api_key}",
        ],
        capture_output=True,
    )
    if curl.returncode == 0:
        typer.echo(f"Instance {_instance_id} destroyed via API.")
    else:
        typer.echo(
            f"WARNING: Failed to destroy instance {_instance_id} — "
            "please remove it manually at https://cloud.vast.ai/"
        )


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))


def _ask(prompt):
    """Wrap questionary .ask() and exit cleanly on Ctrl-C (returns None)."""
    result = prompt.ask()
    if result is None:
        raise SystemExit(1)
    return result


# ── Offer helpers ─────────────────────────────────────────────────────────────

def parse_offers(raw: str) -> list[dict]:
    data = json.loads(raw)
    if isinstance(data, dict):
        return data.get("offers", list(data.values())) if "offers" in data else [data]
    return data


def _fmt_gb(mb) -> str:
    return f"{mb / 1024:.0f}GB" if mb else "-"


def _fmt_mbps(mbps) -> str:
    if not mbps:
        return "-"
    return f"{mbps / 1000:.1f}Gbps" if mbps >= 1000 else f"{mbps:.0f}Mbps"


def _fmt_cost(c) -> str:
    return f"${c:.4f}/GB" if c is not None else "-"


def _fmt_rel(r) -> str:
    return f"{r * 100:.1f}%" if r is not None else "-"


def print_offers_table(offers: list[dict]) -> None:
    sep = "─" * 72
    for i, o in enumerate(offers[:10], 1):
        price = o.get("dph_total", o.get("dph_base", 0))
        gpu = o.get("gpu_name", "unknown")
        n = o.get("num_gpus", 1)
        gpu_str = f"{n}× {gpu}" if n and n > 1 else gpu
        typer.echo(sep)
        typer.echo(f"  #{i}  ID: {o['id']}   {gpu_str}   ${price:.3f}/hr")
        typer.echo(
            f"  Hardware  : VRAM {_fmt_gb(o.get('gpu_ram'))}  |  "
            f"RAM {_fmt_gb(o.get('cpu_ram'))}  |  "
            f"Disk {o.get('disk_space', 0):.0f}GB  |  "
            f"CPU {o.get('cpu_cores_effective', o.get('cpu_cores', '-'))} cores  |  "
            f"PCIe {o.get('pcie_bw', 0):.1f}GB/s"
        )
        typer.echo(
            f"  Location  : {o.get('geolocation') or o.get('country') or '-'}  |  "
            f"Reliability {_fmt_rel(o.get('reliability2'))}  |  "
            f"Static IP: {'yes' if o.get('static_ip') else 'no'}"
        )
        typer.echo(
            f"  Internet  : ↑{_fmt_mbps(o.get('inet_up'))} / ↓{_fmt_mbps(o.get('inet_down'))}  |  "
            f"Upload cost {_fmt_cost(o.get('inet_up_cost'))}  |  "
            f"Download cost {_fmt_cost(o.get('inet_down_cost'))}"
        )
        typer.echo(
            f"  Storage   : {_fmt_cost(o.get('storage_cost'))}/mo  |  "
            f"Driver {o.get('driver_version', '-')}  |  "
            f"CUDA max {o.get('cuda_max_good', o.get('cuda_vers', '-'))}"
        )
        dlperf = f"{o.get('dlperf', 0):.1f}" if o.get("dlperf") else "-"
        perf_d = f"{o.get('dlperf_per_dphtotal', 0):.1f}" if o.get("dlperf_per_dphtotal") else "-"
        typer.echo(f"  DL Perf   : {dlperf} TFLOPS  |  Perf/$ {perf_d}")
    typer.echo(sep)


# ── SSH URL parsing ───────────────────────────────────────────────────────────

def parse_ssh_url(ssh_full: str) -> tuple[str, str]:
    ssh_full = ssh_full.strip()
    if ssh_full.startswith("ssh://"):
        without_scheme = ssh_full[len("ssh://"):]
        host, _, port = without_scheme.rpartition(":")
        return host, port
    # "ssh root@host -p PORT" format
    parts = ssh_full.split()
    host = parts[1]
    port = parts[parts.index("-p") + 1]
    return host, port


# ── Remote runner script template ─────────────────────────────────────────────

def make_remote_runner(
    entrypoint_cmd: str,
    extra_overrides: str,
    keep_alive: bool,
    instance_id: str,
    vastai_api_key: str,
    r2_account_id: str = "",
    r2_access_key: str = "",
    r2_secret_key: str = "",
) -> str:
    keep_alive_str = "true" if keep_alive else "false"
    r2_exports = ""
    if r2_account_id and r2_access_key and r2_secret_key:
        r2_exports = (
            f"export CF_R2_ACCOUNT_ID={r2_account_id}\n"
            f"export CF_R2_ACCESS_KEY={r2_access_key}\n"
            f"export CF_R2_SECRET_KEY={r2_secret_key}\n"
        )
        # inject log path so Python can upload training.log during checkpoints
        extra_overrides = (extra_overrides + " r2_log_path=/workspace/training.log").strip()
    cmd_line = f"{entrypoint_cmd} {extra_overrides}".strip()
    return f"""\
#!/usr/bin/env bash
cd /workspace
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
{r2_exports}
echo "[remote] Starting: {cmd_line}"
{cmd_line}
EXIT_CODE=$?
echo "[remote] Run finished (exit $EXIT_CODE)."
if [[ "{keep_alive_str}" == "true" ]]; then
  echo "[remote] --keep-alive set: instance will NOT be destroyed."
else
  echo "[remote] Waiting 15s for log stream to flush before destroying instance..."
  sleep 15
  echo "[remote] Destroying instance..."
  curl -s -X DELETE "https://console.vast.ai/api/v0/instances/{instance_id}/" \\
    -H "Authorization: Bearer {vastai_api_key}" > /dev/null
  echo "[remote] Instance destroy request sent."
fi
exit $EXIT_CODE
"""


# ── Subprocess helpers ────────────────────────────────────────────────────────

SSH_OPTS = ["-o", "StrictHostKeyChecking=no"]


def run_ssh(host: str, port: str, *cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-p", port, *SSH_OPTS, host, *cmd],
        check=check,
    )


def run_scp(local: str, remote_host: str, remote_path: str, port: str) -> None:
    subprocess.run(
        ["scp", "-P", port, *SSH_OPTS, local, f"{remote_host}:{remote_path}"],
        check=True,
    )


def run_rsync(project_dir: str, remote_host: str, port: str) -> None:
    subprocess.run(
        [
            "rsync", "-az", "--progress",
            "-e", f"ssh -p {port} {' '.join(SSH_OPTS)}",
            "--exclude=.git/",
            "--exclude=.venv/",
            "--exclude=outputs/",
            "--exclude=results/",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--exclude=.python-version",
            "--exclude=.env",
            f"{project_dir}/",
            f"{remote_host}:/workspace/",
        ],
        check=True,
    )


# ── Validators ────────────────────────────────────────────────────────────────

def _validate_price(val: str):
    try:
        if float(val) > 0:
            return True
    except ValueError:
        pass
    return "Please enter a positive number (e.g. 0.50)."


def _validate_offer_choice(max_n: int):
    def _inner(val: str):
        if val.isdigit() and 1 <= int(val) <= max_n:
            return True
        return f"Please enter a number between 1 and {max_n}."
    return _inner


# ── Typer app ─────────────────────────────────────────────────────────────────

app = typer.Typer(add_completion=False)


@app.command()
def train(
    auto: bool = typer.Option(False, "--auto", help="Skip all prompts; auto-select cheapest offer."),
    keep_alive: bool = typer.Option(False, "--keep-alive", help="Do not destroy instance after training."),
) -> None:
    global _instance_id, _instance_started, _vastai_api_key

    # ── Step 0: Preflight ──────────────────────────────────────────────────────
    if subprocess.run(["which", "vastai"], capture_output=True).returncode != 0:
        typer.echo("ERROR: vastai CLI not found.", err=True)
        typer.echo("Install it with: pip install vastai", err=True)
        typer.echo("Then set your API key: vastai set api-key <YOUR_KEY>", err=True)
        raise SystemExit(1)

    typer.echo("Checking Vast.ai authentication...")
    if subprocess.run(["vastai", "show", "user", "--raw"], capture_output=True).returncode != 0:
        typer.echo("ERROR: Vast.ai authentication failed.", err=True)
        typer.echo("Run: vastai set api-key <YOUR_KEY>", err=True)
        raise SystemExit(1)
    typer.echo("Authenticated.\n")

    script_dir = Path(__file__).parent
    env_file = script_dir.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("VAST_API_KEY="):
                _vastai_api_key = line[len("VAST_API_KEY="):].strip()
            elif line.startswith("CF_R2_ACCOUNT_ID="):
                _r2_account_id = line[len("CF_R2_ACCOUNT_ID="):].strip()
            elif line.startswith("CF_R2_ACCESS_KEY="):
                _r2_access_key = line[len("CF_R2_ACCESS_KEY="):].strip()
            elif line.startswith("CF_R2_SECRET_KEY="):
                _r2_secret_key = line[len("CF_R2_SECRET_KEY="):].strip()

    if not _vastai_api_key:
        typer.echo(f"ERROR: VAST_API_KEY not found in {env_file}", err=True)
        typer.echo("Add a line: VAST_API_KEY=<your_key>", err=True)
        raise SystemExit(1)

    os.environ["VAST_API_KEY"] = _vastai_api_key

    # ── Step 1: Configuration prompts ─────────────────────────────────────────
    if auto:
        gpu_filter = GPU_FILTERS[0]
        cuda_min = CUDA_MINS[0]
        docker_image = DOCKER_IMAGES[0]
        entrypoint_cmd = ENTRY_CMDS[0]
        extra_overrides = ""
        max_price = None  # resolved after search
    else:
        gpu_label = _ask(questionary.select("Select GPU type:", choices=GPU_NAMES))
        gpu_idx = GPU_NAMES.index(gpu_label)
        gpu_filter = GPU_FILTERS[gpu_idx]
        typer.echo(f"  → {gpu_label} ({gpu_filter})\n")

        cuda_label = _ask(questionary.select("Select CUDA version:", choices=CUDA_LABELS))
        cuda_idx = CUDA_LABELS.index(cuda_label)
        cuda_min = CUDA_MINS[cuda_idx]
        docker_image = DOCKER_IMAGES[cuda_idx]
        typer.echo(f"  → {cuda_label} (image: {docker_image})\n")

        entry_label = _ask(questionary.select("Select entrypoint:", choices=ENTRY_LABELS))
        entry_idx = ENTRY_LABELS.index(entry_label)
        entrypoint_cmd = ENTRY_CMDS[entry_idx]
        typer.echo(f"  → {entry_label}\n")

        extra_overrides = (_ask(questionary.text(
            "Extra Hydra overrides (optional, e.g. env=HalfCheetah-v5 seed=42):",
            default="",
        )) or "").strip()
        typer.echo("")

        max_price_str = _ask(questionary.text(
            "Max price per hour in USD (e.g. 0.50):",
            validate=_validate_price,
        ))
        max_price = float(max_price_str)
        typer.echo(f"  → ${max_price}/hr max\n")

    # ── Step 2: Search for offers ──────────────────────────────────────────────
    typer.echo("Searching for available offers...")
    price_filter = f"dph<={max_price}" if max_price is not None else "dph<=999"
    result = subprocess.run(
        [
            "vastai", "search", "offers",
            f"gpu_name={gpu_filter} cuda_vers>={cuda_min} {price_filter} rentable=true",
            "-o", "dph", "--raw",
        ],
        capture_output=True,
        text=True,
    )
    offers = parse_offers(result.stdout)

    if not offers:
        typer.echo(
            f"No offers found matching: GPU={gpu_filter}, CUDA>={cuda_min}, "
            + (f"price<=${max_price}/hr" if max_price else ""),
            err=True,
        )
        typer.echo("Try relaxing your constraints.", err=True)
        raise SystemExit(1)

    typer.echo(f"\nTop offers (sorted by price):")
    print_offers_table(offers)
    typer.echo("")

    if auto:
        o = offers[0]
        offer_id = str(o["id"])
        offer_price = o.get("dph_total", o.get("dph_base", 0))
        typer.echo(f"Auto-selecting cheapest offer: #1 (ID: {offer_id}, ${offer_price:.3f}/hr)")
        # also set max_price from cheapest for display
        if max_price is None:
            max_price = offer_price
    else:
        sel_str = _ask(questionary.text(
            f"Enter the number of the offer to use [1-{min(len(offers), 10)}]:",
            validate=_validate_offer_choice(min(len(offers), 10)),
        ))
        sel_idx = int(sel_str) - 1
        o = offers[sel_idx]
        offer_id = str(o["id"])
        offer_price = o.get("dph_total", o.get("dph_base", 0))
        typer.echo(f"  → Selected offer #{sel_str} (ID: {offer_id}, ${offer_price:.3f}/hr)")

    if not auto:
        confirmed = _ask(questionary.confirm(
            f"Create instance (offer {offer_id}) for ${offer_price:.3f}/hr?",
            default=False,
        ))
        if not confirmed:
            typer.echo("Aborted.")
            raise SystemExit(0)
    typer.echo("")

    # ── Step 3: Create instance ────────────────────────────────────────────────
    typer.echo(f"Creating instance from offer {offer_id}...")
    result = subprocess.run(
        [
            "vastai", "create", "instance", offer_id,
            "--image", docker_image,
            "--disk", "50",
            "--ssh", "--raw",
        ],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        typer.echo("ERROR: empty response from vastai create instance.", err=True)
        raise SystemExit(1)
    d = json.loads(result.stdout.strip())
    if d.get("error"):
        typer.echo(f"ERROR: {d.get('msg', 'unknown error')} (status {d.get('status_code', '')})", err=True)
        raise SystemExit(1)
    _instance_id = str(d.get("new_contract") or d.get("id") or "")
    if not _instance_id:
        typer.echo("ERROR: Failed to create instance (no instance ID returned).", err=True)
        raise SystemExit(1)
    typer.echo(f"Instance {_instance_id} created.\n")

    # ── Step 4a: Wait for running ──────────────────────────────────────────────
    typer.echo(f"Waiting for instance to start (timeout: {INSTANCE_START_TIMEOUT // 60} min)...")
    deadline = time.monotonic() + INSTANCE_START_TIMEOUT
    elapsed = 0
    while True:
        r = subprocess.run(["vastai", "show", "instances", "--raw"], capture_output=True, text=True)
        try:
            data = json.loads(r.stdout)
            instances = data.get("instances", data) if isinstance(data, dict) else data
            match = next((i for i in instances if str(i.get("id", "")) == _instance_id), {})
            status = match.get("actual_status", "")
        except Exception:
            status = ""
        if status == "running":
            typer.echo("  Instance is running.\n")
            break
        if time.monotonic() >= deadline:
            typer.echo(f"ERROR: Instance did not start within {INSTANCE_START_TIMEOUT // 60} minutes (status: {status or 'unknown'}).", err=True)
            raise SystemExit(1)
        typer.echo(f"  Status: {status or 'unknown'} ({elapsed}s elapsed)...")
        time.sleep(10)
        elapsed += 10

    # ── Step 4b: Wait for SSH ──────────────────────────────────────────────────
    ssh_raw = subprocess.run(
        ["vastai", "ssh-url", _instance_id], capture_output=True, text=True
    ).stdout.strip()
    remote_host, remote_port = parse_ssh_url(ssh_raw)

    typer.echo(f"Waiting for SSH to become reachable on port {remote_port}...")
    ssh_deadline = time.monotonic() + 120
    ssh_elapsed = 0
    while True:
        r = subprocess.run(
            [
                "ssh", "-p", remote_port, *SSH_OPTS,
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                remote_host, "true",
            ],
            capture_output=True,
        )
        if r.returncode == 0:
            typer.echo("  SSH is ready.\n")
            break
        if time.monotonic() >= ssh_deadline:
            typer.echo(f"ERROR: SSH did not become reachable within 120s.", err=True)
            raise SystemExit(1)
        typer.echo(f"  SSH not ready yet ({ssh_elapsed}s elapsed)...")
        time.sleep(5)
        ssh_elapsed += 5

    # ── Step 5: Upload code ────────────────────────────────────────────────────
    typer.echo("Uploading project files to instance...")
    project_dir = str(script_dir.parent.resolve())
    run_rsync(project_dir, remote_host, remote_port)
    typer.echo("Upload complete.\n")

    # ── Step 6: Install system libs + Python deps ──────────────────────────────
    typer.echo("Installing system libraries and Python dependencies...")
    setup_script = (
        "set -e\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "apt-get update -qq\n"
        "apt-get install -y --no-install-recommends "
        "libgl1 libglib2.0-0 libgles2 libegl1 libegl-mesa0\n"
        "cd /workspace\n"
        "pip install -q uv\n"
        "uv sync\n"
    )
    subprocess.run(
        ["ssh", "-p", remote_port, *SSH_OPTS, remote_host, "bash"],
        input=setup_script,
        text=True,
        check=True,
    )
    typer.echo("Dependencies installed.\n")

    # ── Step 7: Generate and upload remote_run.sh ──────────────────────────────
    runner_content = make_remote_runner(
        entrypoint_cmd, extra_overrides, keep_alive, _instance_id, _vastai_api_key,
        r2_account_id=_r2_account_id, r2_access_key=_r2_access_key, r2_secret_key=_r2_secret_key,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(runner_content)
        tmp_runner = f.name
    try:
        run_scp(tmp_runner, remote_host, "/workspace/remote_run.sh", remote_port)
    finally:
        os.unlink(tmp_runner)
    typer.echo("Remote runner uploaded.\n")

    # ── Step 8: Launch training (nohup) ───────────────────────────────────────
    typer.echo("Launching training (nohup)...")
    run_ssh(
        remote_host, remote_port,
        "chmod +x /workspace/remote_run.sh && "
        "nohup bash /workspace/remote_run.sh > /workspace/training.log 2>&1 &",
    )
    _instance_started = True  # disarm cleanup trap
    typer.echo(f"Training running on instance {_instance_id}.")
    typer.echo("Instance will self-destruct when training completes.\n")

    # ── Step 9: Stream logs ───────────────────────────────────────────────────
    typer.echo("Streaming live logs (Ctrl-C to detach — training continues server-side):")
    typer.echo("════════════════════════════════════════════════════════════════════════")
    try:
        subprocess.run(
            [
                "ssh", "-p", remote_port, *SSH_OPTS, remote_host,
                "until [ -f /workspace/training.log ]; do sleep 1; done; "
                "tail -f /workspace/training.log",
            ]
        )
    except KeyboardInterrupt:
        pass
    typer.echo("\nDetached from log stream.")
    typer.echo(f"Instance {_instance_id} is still running and will self-destruct when done.")
    typer.echo(f"To reconnect:  ssh -p {remote_port} {' '.join(SSH_OPTS)} {remote_host}")


if __name__ == "__main__":
    app()
