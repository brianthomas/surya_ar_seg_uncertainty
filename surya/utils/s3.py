"""Helpers for reading SDO NetCDF data from S3.

Used when `sdo_data_root_path` (or an index `path` column) points at an S3 URI
rather than a local directory. Only boto3 is required; the fsspec/s3fs stack is
deliberately not used, because HDF5 needs random seeks and streaming one 4096x4096
13-channel file takes ~60s versus ~2s to download it whole.
"""

import os
import tempfile

# Optional: the helpers below degrade to a clear ImportError when boto3 is absent,
# so the package still imports in environments that never touch S3.
try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except Exception:  # pragma: no cover
    boto3 = None
    TransferConfig = None
    UNSIGNED = None
    BotoConfig = None

    class ClientError(Exception):
        """Placeholder so `except ClientError` is valid without botocore installed."""

SCRATCH_DIR_ENV_VARS = ("SURYA_S3_SCRATCH", "SCRATCH", "TMPDIR")

# Error codes that mean "these credentials cannot read this object", as opposed to
# a transient fault worth retrying.
S3_DENIED_CODES = frozenset(
    {"403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"}
)


def is_s3_path(path) -> bool:
    """Return True if `path` is an S3 URI (starts with ``s3://``)."""
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    Raises:
        ValueError: If `uri` is not an S3 URI, or has no key component.
    """
    if not is_s3_path(uri):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    remainder = uri[len("s3://"):]
    if "/" not in remainder:
        raise ValueError(f"S3 URI is missing a key: {uri!r} (expected s3://bucket/key)")
    bucket, key = remainder.split("/", 1)
    return bucket, key


def default_scratch_dir() -> str:
    """Return the directory used to stage S3 downloads.

    Files are deleted immediately after being read, so this needs room for
    `num_data_workers` x `world_size` x ~0.6 GB, not for the whole dataset.
    Prefers node-local scratch over the home filesystem.
    """
    for env_var in SCRATCH_DIR_ENV_VARS:
        root = os.environ.get(env_var)
        if root:
            return os.path.join(root, "surya_s3_scratch")
    return os.path.join(tempfile.gettempdir(), "surya_s3_scratch")


def make_s3_client(
    anon: bool = False,
    region: str | None = None,
    pool_size: int = 32,
    max_attempts: int = 10,
):
    """Return a configured boto3 S3 client.

    Args:
        anon: If True, send unsigned requests (public buckets).
        region: AWS region. Pass ``None`` to let boto3 resolve it from the
            environment or the instance metadata service.
        pool_size: Max connections in the client's pool. Should be at least twice
            the download concurrency, or multipart threads contend for connections.
        max_attempts: Retry attempts in adaptive mode. Set high, because a failed
            read kills a training run.
    """
    if boto3 is None:
        raise ImportError("boto3 is required for S3 access. Install via: pip install boto3")

    retry_cfg = {"max_attempts": max_attempts, "mode": "adaptive"}
    if anon:
        config = BotoConfig(
            signature_version=UNSIGNED,
            max_pool_connections=pool_size,
            retries=retry_cfg,
        )
    else:
        config = BotoConfig(max_pool_connections=pool_size, retries=retry_cfg)
    return boto3.client("s3", region_name=region, config=config)
