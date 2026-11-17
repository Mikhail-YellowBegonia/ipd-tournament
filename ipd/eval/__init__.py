"""评估协议：行为指纹去重与多配比评估（issue #2）。"""

from ipd.eval.fingerprint import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_SEEDS,
    Fingerprint,
    classify_clusters,
    classify_fingerprint,
    cluster_fingerprints,
    cluster_weights,
    dedupe,
    distance,
    fingerprint,
    fingerprint_many,
)

__all__ = [
    "DEFAULT_CLUSTER_THRESHOLD",
    "DEFAULT_SEEDS",
    "Fingerprint",
    "classify_clusters",
    "classify_fingerprint",
    "cluster_fingerprints",
    "cluster_weights",
    "dedupe",
    "distance",
    "fingerprint",
    "fingerprint_many",
]
