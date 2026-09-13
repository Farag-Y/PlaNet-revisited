#!/usr/bin/env python3
"""Clean up stale checkpoints and experience replays in the R2 bucket.

For each run prefix (a date_timestamp "folder" produced by a training run), this keeps only:
  - the highest-numbered checkpoint_{episode}/ dir
  - the highest-numbered experience_replay_{episode}.pt file
and deletes every other checkpoint dir / experience replay under that prefix.

It never touches config.yaml, metrics.png, or training.log.

Usage:
    uv run python scripts/r2_cleanup.py                       # scan every run prefix in the bucket
    uv run python scripts/r2_cleanup.py --prefix 2026-09-10_14-23-01   # scan a single run
    uv run python scripts/r2_cleanup.py --bucket my-bucket
    uv run python scripts/r2_cleanup.py --yes                 # skip the confirmation prompt
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cloud_storage

DEFAULT_BUCKET = "planet-checkpoints-vastai"


def _load_env_file() -> None:
    """Load CF_R2_* credentials from the repo-root .env, without overriding already-set env vars."""
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}PB"


def plan_deletions(client, bucket: str, prefix: str) -> list[dict]:
    """Return the list of {Key, Size} objects to delete for a single run prefix."""
    to_delete: list[dict] = []

    checkpoints = cloud_storage.list_checkpoint_episodes(client, bucket, prefix)
    if checkpoints:
        keep_episode = max(checkpoints)
        for episode, objects in checkpoints.items():
            if episode != keep_episode:
                to_delete.extend(objects)

    replays = cloud_storage.list_experience_replay_episodes(client, bucket, prefix)
    if replays:
        keep_episode = max(replays)
        for episode, obj in replays.items():
            if episode != keep_episode:
                to_delete.append(obj)

    return to_delete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefix", "--date", dest="prefix", default=None,
                         help="Only clean this run prefix (date_timestamp). Default: all prefixes in the bucket.")
    parser.add_argument("--bucket", default=None,
                         help=f"R2 bucket name. Default: $R2_BUCKET or '{DEFAULT_BUCKET}'.")
    parser.add_argument("--yes", "-y", action="store_true",
                         help="Skip the confirmation prompt and delete immediately.")
    args = parser.parse_args()

    _load_env_file()
    for var in ("CF_R2_ACCOUNT_ID", "CF_R2_ACCESS_KEY", "CF_R2_SECRET_KEY"):
        if var not in os.environ:
            print(f"ERROR: {var} not set (add it to .env or export it).", file=sys.stderr)
            raise SystemExit(1)

    bucket = args.bucket or os.environ.get("R2_BUCKET") or DEFAULT_BUCKET
    client = cloud_storage.get_client()

    prefixes = [args.prefix] if args.prefix else cloud_storage.list_run_prefixes(client, bucket)
    if not prefixes:
        print(f"No run prefixes found in bucket '{bucket}'.")
        return

    print(f"Bucket: {bucket}")
    print(f"Scanning {len(prefixes)} run prefix(es): {', '.join(prefixes)}\n")

    plan: dict[str, list[dict]] = {}
    for prefix in prefixes:
        deletions = plan_deletions(client, bucket, prefix)
        if deletions:
            plan[prefix] = deletions

    if not plan:
        print("Nothing to delete — every prefix already keeps only its latest checkpoint and experience replay.")
        return

    total_keys = 0
    total_size = 0
    for prefix, deletions in plan.items():
        print(f"[{prefix}]")
        for obj in sorted(deletions, key=lambda o: o["Key"]):
            print(f"  - {obj['Key']}  ({_human_size(obj['Size'])})")
        total_keys += len(deletions)
        total_size += sum(obj["Size"] for obj in deletions)
        print()

    print(f"Total: {total_keys} object(s), {_human_size(total_size)}, across {len(plan)} run prefix(es).\n")

    if not args.yes:
        answer = input("Delete these objects? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted — nothing was deleted.")
            return

    all_keys = [obj["Key"] for deletions in plan.values() for obj in deletions]
    cloud_storage.delete_keys(client, bucket, all_keys)
    print(f"Deleted {len(all_keys)} object(s).")


if __name__ == "__main__":
    main()
