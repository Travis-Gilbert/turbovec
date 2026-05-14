"""Phase 1 parity tests for the turbovec.mlx Apple/Metal backend.

These cover the Rust -> Python -> MLX bridge only:
  * The Rust-supplied rotation matrix is actually orthogonal.
  * MLX matmul agrees with numpy matmul on the same R and same input
    (to within fp32 reduction-order noise).
  * The Lloyd-Max codebook has the right shape and monotonic boundaries.

They do NOT cover end-to-end encode/search correctness — that arrives
with the Metal kernels in phases 2 and 3.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from turbovec._turbovec import codebook as rust_codebook
from turbovec._turbovec import make_rotation_matrix as rust_make_rotation_matrix
from turbovec.mlx import TurboQuantIndex


@pytest.mark.parametrize("dim", [64, 200, 384, 1536])
def test_rust_rotation_matrix_is_orthogonal(dim):
    R = rust_make_rotation_matrix(dim)
    assert R.shape == (dim, dim)
    assert R.dtype == np.float32

    identity = R @ R.T
    np.testing.assert_allclose(identity, np.eye(dim, dtype=np.float32), atol=1e-4)


@pytest.mark.parametrize("dim", [64, 200, 1536])
def test_mlx_rotation_matches_numpy(dim):
    R = rust_make_rotation_matrix(dim)
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8, dim)).astype(np.float32)
    x_np /= np.linalg.norm(x_np, axis=1, keepdims=True)

    expected = x_np @ R.T

    index = TurboQuantIndex(dim=dim, bit_width=4)
    x_mx = mx.array(x_np)
    actual = np.asarray(index._rotate(x_mx))

    np.testing.assert_allclose(actual, expected, atol=1e-4)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 1536])
def test_codebook_shapes_and_monotonic(bit_width, dim):
    boundaries, centroids = rust_codebook(bit_width, dim)
    n_levels = 1 << bit_width

    assert boundaries.shape == (n_levels - 1,)
    assert centroids.shape == (n_levels,)
    assert np.all(np.diff(boundaries) > 0), "boundaries must be strictly increasing"
    assert np.all(np.diff(centroids) > 0), "centroids must be strictly increasing"


def test_index_construction_rejects_bad_bit_width():
    with pytest.raises(ValueError):
        TurboQuantIndex(dim=64, bit_width=3)


def test_search_empty_index_returns_empty():
    index = TurboQuantIndex(dim=64, bit_width=4)
    scores, indices = index.search(np.zeros((2, 64), dtype=np.float32), k=10)
    assert scores.shape == (2, 0)
    assert indices.shape == (2, 0)
    assert scores.dtype == np.float32
    assert indices.dtype == np.int64


def _random_unit_vectors(n, dim, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return v


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_add_norms_match_rust(dim, bit_width):
    from turbovec._turbovec import encode as rust_encode

    vectors = _random_unit_vectors(32, dim, seed=0)
    rust_packed, rust_norms = rust_encode(vectors, bit_width)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors)
    mlx_norms = np.asarray(index._norms)

    np.testing.assert_allclose(mlx_norms, rust_norms, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_add_codes_match_rust(dim, bit_width):
    """Byte-exact parity vs the Rust CPU encode.

    In principle, tiny float drift between Accelerate's GEMM and MLX's
    Metal matmul could flip codes for coordinates within ~1e-6 of a
    Lloyd-Max boundary. In practice we observe zero drift across the
    configs tested here — assert bit-exact equality, and we'll relax if
    a future config ever flakes.
    """
    from turbovec._turbovec import encode as rust_encode

    vectors = _random_unit_vectors(32, dim, seed=1)
    rust_packed, _ = rust_encode(vectors, bit_width)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors)
    mlx_packed = np.asarray(index._packed_codes)

    assert mlx_packed.shape == rust_packed.shape
    assert mlx_packed.dtype == rust_packed.dtype
    assert np.array_equal(mlx_packed, rust_packed), (
        f"byte-level parity failed: "
        f"{np.unpackbits(rust_packed ^ mlx_packed).sum()} bits differ"
    )


def test_add_accumulates_across_calls():
    dim, bit_width = 128, 4
    vectors_a = _random_unit_vectors(10, dim, seed=2)
    vectors_b = _random_unit_vectors(7, dim, seed=3)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(vectors_a)
    index.add(vectors_b)
    assert len(index) == 17
    assert index._packed_codes.shape == (17, bit_width * dim // 8)
    assert index._norms.shape == (17,)


def _numpy_scores(packed, centroids, norms, q_rot, bit_width, dim):
    """Pure-numpy reference scorer mirroring the MLX kernel.

    Decodes the bit-plane packed bytes back to integer codes, looks up
    centroid values, dot-products with the rotated query, and scales by
    norms. Used as a precision-independent oracle for the MLX scoring
    kernel — does not match the Rust u8-LUT path exactly.
    """
    n_db, bytes_per_vec = packed.shape
    plane_size = dim // 8
    codes = np.zeros((n_db, dim), dtype=np.uint8)
    bit_pos = 7 - (np.arange(dim) % 8)
    byte_pos = np.arange(dim) // 8
    for p in range(bit_width):
        plane = packed[:, p * plane_size:(p + 1) * plane_size]
        bits = (plane[:, byte_pos] >> bit_pos) & 1
        codes |= bits.astype(np.uint8) << p
    decoded = centroids[codes]                       # (n_db, dim)
    return (q_rot @ decoded.T) * norms[None, :]      # (nq, n_db)


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_search_matches_numpy_oracle(dim, bit_width):
    """The MLX scoring kernel matches a pure-numpy reference.

    Independent of the Rust path — proves the kernel computes
    ``sum_j q_rot[j] * centroids[code_j] * norms[v]`` correctly.
    """
    from turbovec._turbovec import codebook as rust_codebook
    from turbovec._turbovec import make_rotation_matrix as rust_make_rotation_matrix

    n_db, n_q, k = 64, 4, 8
    db = _random_unit_vectors(n_db, dim, seed=30)
    queries = _random_unit_vectors(n_q, dim, seed=31)

    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(db)
    mlx_scores, mlx_idx = index.search(queries, k=k)

    R = rust_make_rotation_matrix(dim)
    _, centroids = rust_codebook(bit_width, dim)
    q_rot = queries @ R.T
    packed = np.asarray(index._packed_codes)
    norms = np.asarray(index._norms)
    full_scores = _numpy_scores(packed, centroids, norms, q_rot, bit_width, dim)

    expected_idx = np.argsort(-full_scores, axis=1)[:, :k]
    expected_scores = np.take_along_axis(full_scores, expected_idx, axis=1)

    # Returned top-k set matches numpy oracle exactly (no precision
    # divergence — MLX uses the same fp32 dequantize-and-dot path).
    for q in range(n_q):
        assert set(mlx_idx[q].tolist()) == set(expected_idx[q].tolist())
    np.testing.assert_allclose(
        np.sort(mlx_scores, axis=1),
        np.sort(expected_scores, axis=1),
        atol=1e-3,
    )


@pytest.mark.parametrize("bit_width", [2, 4])
@pytest.mark.parametrize("dim", [64, 128, 1536])
def test_search_recall_vs_rust(dim, bit_width):
    """Recall@k against the Rust CPU search.

    The Rust path runs u8-LUT FastScan; MLX runs straight fp32
    dequantize-and-dot. Both compute the same TurboQuant scoring
    function but with different precision, so the top-k boundary can
    swap when consecutive scores are tied within u8-LUT rounding.
    We require recall@k >= 0.9 (typically 1.0 in the interior and
    occasionally drops a single boundary slot).
    """
    from turbovec import TurboQuantIndex as RustIndex

    n_db, n_q, k = 256, 8, 10
    db = _random_unit_vectors(n_db, dim, seed=10)
    queries = _random_unit_vectors(n_q, dim, seed=11)

    rust = RustIndex(dim=dim, bit_width=bit_width)
    rust.add(db)
    _, rust_idx = rust.search(queries, k=k)

    mlx_index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    mlx_index.add(db)
    _, mlx_idx = mlx_index.search(queries, k=k)

    recalls = [
        len(set(rust_idx[q].tolist()) & set(mlx_idx[q].tolist())) / k
        for q in range(n_q)
    ]
    mean_recall = sum(recalls) / n_q
    assert mean_recall >= 0.9, f"mean recall@{k} = {mean_recall:.3f}, recalls={recalls}"


def test_search_returns_fewer_than_k_when_index_small():
    dim, bit_width = 64, 4
    index = TurboQuantIndex(dim=dim, bit_width=bit_width)
    index.add(_random_unit_vectors(3, dim, seed=20))
    scores, idx = index.search(_random_unit_vectors(2, dim, seed=21), k=10)
    assert scores.shape == (2, 3)
    assert idx.shape == (2, 3)
