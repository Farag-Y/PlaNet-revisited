import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_client():
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
    client = get_client()
    key = f"{prefix}/config.yaml"
    print(f"[R2] Uploading config → {key}")
    client.put_object(
        Bucket=cfg.r2_bucket,
        Key=key,
        Body=OmegaConf.to_yaml(cfg).encode(),
    )


def upload_checkpoint(cfg: DictConfig, checkpoint_dir: str, episode: int, prefix: str) -> None:
    client = get_client()
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


def upload_experience_replay(cfg: DictConfig, local_path: str, episode: int, prefix: str) -> None:
    client = get_client()
    key = f"{prefix}/experience_replay_{episode}.pt"
    _upload_file(client, local_path, cfg.r2_bucket, key)


def download_checkpoint(cfg: DictConfig, episode: int, dest_dir: str, prefix: str) -> None:
    client = get_client()
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


def list_run_prefixes(client, bucket: str) -> list[str]:
    """Top-level run prefixes in the bucket (the date_timestamp 'folders')."""
    prefixes = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            prefixes.append(common_prefix["Prefix"].rstrip("/"))
    return sorted(prefixes)


def list_checkpoint_episodes(client, bucket: str, prefix: str) -> dict[int, list[dict]]:
    """Map episode -> list of {Key, Size} objects, for every checkpoint_{episode}/ dir under prefix."""
    episodes: dict[int, list[dict]] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/checkpoint_"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rest = key[len(f"{prefix}/checkpoint_"):]
            episode_str = rest.split("/", 1)[0]
            if not episode_str.isdigit():
                continue
            episodes.setdefault(int(episode_str), []).append({"Key": key, "Size": obj["Size"]})
    return episodes


def list_experience_replay_episodes(client, bucket: str, prefix: str) -> dict[int, dict]:
    """Map episode -> {Key, Size}, for every experience_replay_{episode}.pt file under prefix."""
    episodes: dict[int, dict] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/experience_replay_"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            name = key[len(f"{prefix}/"):]
            if not (name.startswith("experience_replay_") and name.endswith(".pt")):
                continue
            episode_str = name[len("experience_replay_"):-len(".pt")]
            if not episode_str.isdigit():
                continue
            episodes[int(episode_str)] = {"Key": key, "Size": obj["Size"]}
    return episodes


def delete_keys(client, bucket: str, keys: list[str]) -> None:
    """Batch-delete keys (S3 delete_objects caps at 1000 keys per call)."""
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch]},
        )
