import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _get_client(cfg: DictConfig):
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['CF_R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["CF_R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["CF_R2_SECRET_KEY"],
        region_name="auto",
    )


def _upload_file(client, local_path: str, bucket: str, key: str) -> None:
    print(f"[R2] {Path(local_path).name} → {key}")
    client.upload_file(local_path, bucket, key)


def upload_config(cfg: DictConfig, prefix: str) -> None:
    client = _get_client(cfg)
    key = f"{prefix}/config.yaml"
    print(f"[R2] Uploading config → {key}")
    client.put_object(
        Bucket=cfg.r2_bucket,
        Key=key,
        Body=OmegaConf.to_yaml(cfg).encode(),
    )


def upload_checkpoint(cfg: DictConfig, checkpoint_dir: str, episode: int, prefix: str) -> None:
    client = _get_client(cfg)
    results_dir = Path(checkpoint_dir).parent

    # Per-checkpoint files (versioned); skip the replay buffer — it's large and not needed remotely
    for file in Path(checkpoint_dir).iterdir():
        if file.name == "experience_replay.pt":
            continue
        _upload_file(client, str(file), cfg.r2_bucket, f"{prefix}/checkpoint_{episode}/{file.name}")

    # Single-file overwrite: metrics plot
    metrics_png = results_dir / "metrics.png"
    if metrics_png.exists():
        _upload_file(client, str(metrics_png), cfg.r2_bucket, f"{prefix}/metrics.png")

    # Single-file overwrite: training log (Vast.ai only)
    log_path = getattr(cfg, "r2_log_path", "")
    if log_path and Path(log_path).exists():
        _upload_file(client, log_path, cfg.r2_bucket, f"{prefix}/training.log")

    print(f"[R2] Checkpoint {episode} uploaded under {prefix}/")


def download_checkpoint(cfg: DictConfig, episode: int, dest_dir: str, prefix: str) -> None:
    client = _get_client(cfg)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    key_prefix = f"{prefix}/checkpoint_{episode}/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg.r2_bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest_file = dest / Path(key).name
            print(f"[R2] {key} → {dest_file}")
            client.download_file(cfg.r2_bucket, key, str(dest_file))
    print(f"[R2] Checkpoint {episode} downloaded to {dest_dir}.")
