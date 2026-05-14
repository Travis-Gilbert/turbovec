"""Apple GPU (Metal via MLX) backend for turbovec.

Provides :class:`TurboQuantIndex` running on Apple Silicon GPUs through
MLX. The rotation matrix and Lloyd-Max codebook are sourced from the
Rust crate (``_turbovec.make_rotation_matrix`` /
``_turbovec.codebook``), so ``.tv`` / ``.tvim`` files written by this
backend round-trip bit-exactly with the CPU index.

Phases:
    1. Rotation parity + scaffold (current).
    2. Encode kernel — fused rotate + Lloyd-Max quantize + bit-pack.
    3. Search kernel — fused LUT-build + nibble-scan + top-k.
    4. ``.tv`` / ``.tvim`` load/save + benchmark harness row.
"""
from __future__ import annotations

try:
    import mlx.core as mx
except ImportError as e:
    raise ImportError(
        "turbovec.mlx requires the 'mlx' package. "
        "Install with: pip install 'turbovec[mlx]'"
    ) from e

import numpy as np

from . import _kernels
from .._turbovec import codebook as _rust_codebook
from .._turbovec import make_rotation_matrix as _rust_make_rotation_matrix


__all__ = ["TurboQuantIndex"]


class TurboQuantIndex:
    """TurboQuant vector index running on Apple GPU via MLX.

    Mirrors the API of :class:`turbovec.TurboQuantIndex` but executes
    the rotate / quantize / search hot loops as Metal kernels through
    MLX. Currently scaffolding only — ``add`` and ``search`` raise
    ``NotImplementedError`` until the encode and search kernels land
    (phases 2–3).
    """

    def __init__(self, dim: int, bit_width: int) -> None:
        if bit_width not in (2, 4):
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width}")
        if dim % 8 != 0:
            raise ValueError(f"dim must be a multiple of 8, got {dim}")
        self._dim = dim
        self._bit_width = bit_width
        self._n = 0
        self._bytes_per_vec = bit_width * dim // 8

        rotation_np = _rust_make_rotation_matrix(dim)
        boundaries_np, centroids_np = _rust_codebook(bit_width, dim)
        self._rotation = mx.array(rotation_np)
        self._boundaries = mx.array(boundaries_np)
        self._centroids = mx.array(centroids_np)

        self._quantize_pack = _kernels.build_quantize_pack_kernel(dim, bit_width)
        self._score = _kernels.build_score_kernel(dim, bit_width)
        self._packed_codes: "mx.array | None" = None
        self._norms: "mx.array | None" = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def bit_width(self) -> int:
        return self._bit_width

    def __len__(self) -> int:
        return self._n

    def _rotate(self, vectors: "mx.array") -> "mx.array":
        """Apply the shared rotation: ``vectors @ R.T``.

        ``vectors`` is ``(n, dim)`` row-major; result is ``(n, dim)``.
        """
        return vectors @ self._rotation.T

    def add(self, vectors) -> None:
        """Encode ``vectors`` and append to the index.

        ``vectors`` may be a numpy array or an ``mx.array`` of shape
        ``(n, dim)``, dtype ``float32``.
        """
        if not isinstance(vectors, mx.array):
            vectors = mx.array(np.ascontiguousarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"expected shape (n, {self._dim}), got {tuple(vectors.shape)}"
            )
        n = vectors.shape[0]
        if n == 0:
            return

        norms = mx.linalg.norm(vectors, axis=1, stream=mx.default_stream(mx.default_device()))
        safe = mx.maximum(norms, mx.array(1e-10, dtype=mx.float32))
        unit = vectors / safe[:, None]
        rotated = unit @ self._rotation.T
        packed = self._quantize_pack(rotated, self._boundaries)

        if self._packed_codes is None:
            self._packed_codes = packed
            self._norms = norms
        else:
            self._packed_codes = mx.concatenate([self._packed_codes, packed], axis=0)
            self._norms = mx.concatenate([self._norms, norms], axis=0)
        self._n += n

    def search(self, queries, k: int):
        """Return the top-``k`` ``(scores, indices)`` for each query.

        ``queries`` may be a numpy array or an ``mx.array`` of shape
        ``(nq, dim)``, dtype ``float32``. Returns numpy arrays of shape
        ``(nq, effective_k)`` where ``effective_k = min(k, len(index))``,
        with dtypes ``float32`` and ``int64`` respectively — matching
        the CPU :meth:`turbovec.TurboQuantIndex.search` signature.
        """
        if not isinstance(queries, mx.array):
            queries = mx.array(np.ascontiguousarray(queries, dtype=np.float32))
        if queries.ndim != 2 or queries.shape[1] != self._dim:
            raise ValueError(
                f"expected shape (nq, {self._dim}), got {tuple(queries.shape)}"
            )
        nq = queries.shape[0]

        if self._packed_codes is None or self._n == 0:
            return (
                np.zeros((nq, 0), dtype=np.float32),
                np.zeros((nq, 0), dtype=np.int64),
            )

        effective_k = min(k, self._n)
        q_rot = queries @ self._rotation.T
        scores = self._score(q_rot, self._packed_codes, self._centroids, self._norms)

        idx = mx.argsort(-scores, axis=1)[:, :effective_k]
        top_scores = mx.take_along_axis(scores, idx, axis=1)

        return (
            np.asarray(top_scores),
            np.asarray(idx).astype(np.int64),
        )
