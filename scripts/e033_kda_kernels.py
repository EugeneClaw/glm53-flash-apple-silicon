#!/usr/bin/env python3
"""E-033 (P5): KDA glue fused JIT kernels — l2norm-pair + o_norm (RMSNormGated).
Numerical-equivalence harness against the vendored eager paths, then slope-fit
microbench per #3939 (K instances in one mx.eval, T(K)=c+K*s).
No GPU work yet beyond this standalone script (server untouched)."""
import mlx.core as mx

# ---------- fused kernels ----------

_l2norm_pair_kernel = mx.fast.metal_kernel(
    name="kda_l2norm_pair",
    inputs=["q_in", "k_in", "rsqrt_scale"],
    outputs=["q_out", "k_out"],
    source="""
    uint3 pos = thread_position_in_grid2d;
    uint row = pos.x;      // over B*S*H rows
    uint col = pos.y;      // over D (head_dim)
    uint D = q_out.get_shape()[1];

    threadgroup float partial[128];
    float qv = q_in[row * D + col];
    float kv = k_in[row * D + col];
    partial[thread_index_in_threadgroup] = qv * qv;
    threadgroup_barrier();
    // tree-reduce within threadgroup (threadgroup size == D assumed = 128 lanes via simd)
    for (uint offset = threadgroup_size.x >> 1; offset > 0; offset >>= 1) {
        if (thread_index_in_threadgroup < offset) {
            partial[thread_index_in_threadgroup] += partial[thread_index_in_threadgroup + offset];
        }
        threadgroup_barrier();
    }
    float q_sq = partial[0];

    partial[thread_index_in_threadgroup] = kv * kv;
    threadgroup_barrier();
    for (uint offset = threadgroup_size.x >> 1; offset > 0; offset >>= 1) {
        if (thread_index_in_threadgroup < offset) {
            partial[thread_index_in_threadgroup] += partial[thread_index_in_threadgroup + offset];
        }
        threadgroup_barrier();
    }
    float k_sq = partial[0];

    const float eps = 1e-6f;
    float rsqrt_q = rsqrt(q_sq + eps);
    float rsqrt_k = rsqrt(k_sq + eps);
    q_out[row * D + col] = qv * rsqrt_q * rsqrt_scale;
    k_out[row * D + col] = kv * rsqrt_k;
    """,
)

_onorm_kernel = mx.fast.metal_kernel(
    name="kda_onorm_gated",
    inputs=["x_in", "gate_in", "weight"],
    outputs=["out"],
    source="""
    uint3 pos = thread_position_in_grid2d;
    uint row = pos.x;
    uint col = pos.y;
    uint D = out.get_shape()[1];

    threadgroup float partial[128];
    float xv = x_in[row * D + col];
    partial[thread_index_in_threadgroup] = xv * xv;
    threadgroup_barrier();
    for (uint offset = threadgroup_size.x >> 1; offset > 0; offset >>= 1) {
        if (thread_index_in_threadgroup < offset) {
            partial[thread_index_in_threadgroup] += partial[thread_index_in_threadgroup + offset];
        }
        threadgroup_barrier();
    }
    float mean_sq = partial[0] / (float)D;
    const float eps = 1e-6f;
    float normed = xv * rsqrt(mean_sq + eps) * weight[col];
    out[row * D + col] = normed * sigmoid(gate_in[row * D + col]);
    """,
)


def l2norm_pair_fused(q, k, scale):
    """q,k: [N, 128] fp32 (rows = B*S*H). Returns qn (scaled), kn."""
    qf = q.reshape(-1, q.shape[-1])
    kf = k.reshape(-1, k.shape[-1])
    qn, kn = _l2norm_pair_kernel(
        qf, kf, mx.array(scale, dtype=mx.float32),
        grid=(qf.shape[0], qf.shape[1], 1),
        threadgroup_size=(qf.shape[1], 1, 1),
    )
    return qn, kn


def onorm_fused(x, gate, weight):
    xf = x.reshape(-1, x.shape[-1])
    gf = gate.reshape(-1, gate.shape[-1])
    out = _onorm_kernel(
        xf, gf, weight,
        grid=(xf.shape[0], xf.shape[1], 1),
        threadgroup_size=(xf.shape[1], 1, 1),
    )
    return out.reshape(x.shape)


# ---------- eager references (verbatim from vendored language.py) ----------

def _l2norm(x, eps=1e-6):
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def onorm_eager(x, gate, weight, eps=1e-6):
    dt = x.dtype
    xf = x.astype(mx.float32)
    var = (xf * xf).mean(-1, keepdims=True)
    xf = xf * mx.rsqrt(var + eps)
    xf = weight.astype(mx.float32) * xf
    xf = xf * mx.sigmoid(gate.astype(mx.float32))
    return xf.astype(dt)


# ---------- harness ----------

def slope_fit(fn, ks):
    """Time fn() K times inside one eval; return (c, s, r2)."""
    import time as _t
    times = []
    for K in ks:
        outs = [fn() for _ in range(K)]
        mx.eval(outs)
        t0 = _t.perf_counter()
        mx.eval(outs)  # second eval on materialized results = cheap; time build+eval
        # actually: build K graphs fresh, single eval:
        outs = [fn() for _ in range(K)]
        t0 = _t.perf_counter()
        mx.eval(outs)
        times.append((_t.perf_counter() - t0) * 1e3)
    # least squares
    n = len(ks)
    xs = ks
    ys = times
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    s = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    c = (sy - s * sx) / n
    ybar = sy / n
    ssr = sum((y - (c + s * x)) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - ybar) ** 2 for y in ys)
    r2 = 1 - ssr / sst if sst > 0 else 1.0
    return c, s, r2, ys


def main():
    import time as _t
    H, D, S = 32, 128, 2048
    rng = mx.random.key(0)
    q = mx.random.normal((S, H, D), key=rng) * 0.5
    k = mx.random.normal((S, H, D), key=rng) * 0.5
    scale = D ** -0.5
    weight = mx.random.normal((D,), key=rng) * 0.1 + 1.0

    # --- equivalence ---
    q_e = (_l2norm(q.astype(mx.float32)) * scale)
    k_e = _l2norm(k.astype(mx.float32))
    q_f, k_f = l2norm_pair_fused(q.astype(mx.float32), k.astype(mx.float32), scale)
    mx.eval(q_e, k_e, q_f, k_f)
    dq = float(mx.max(mx.abs(q_e - q_f)))
    dk = float(mx.max(mx.abs(k_e - k_f)))
    print(f"l2norm-pair: max|dq|={dq:.3e} max|dk|={dk:.3e}")
    assert dq < 2e-6 and dk < 2e-6, "l2norm equivalence FAILED"

    x = mx.random.normal((S, H, D), key=rng) * 0.7
    g = mx.random.normal((S, H, D), key=rng) * 0.3
    o_e = onorm_eager(x, g, weight)
    o_f = onorm_fused(x.astype(mx.float32), g.astype(mx.float32), weight.astype(mx.float32))
    mx.eval(o_e, o_f)
    do = float(mx.max(mx.abs(o_e.astype(mx.float32) - o_f.astype(mx.float32))))
    print(f"o_norm-gated: max|d|={do:.3e}")
    assert do < 2e-6, "onorm equivalence FAILED"

    # --- slope bench (per #3939) ---
    qf_in = q.astype(mx.float32).reshape(-1, D)
    kf_in = k.astype(mx.float32).reshape(-1, D)

    def eager_pair():
        return (_l2norm(qf_in) * scale, _l2norm(kf_in))

    def fused_pair():
        return l2norm_pair_fused(qf_in, kf_in, scale)

    ks = [2, 4, 8, 16]
    c, s, r2, _ = slope_fit(eager_pair, ks)
    print(f"l2norm eager : {s*1e3:.1f} us/call (floor {c:.2f}ms, R2 {r2:.4f})")
    c2, s2, r22, _ = slope_fit(fused_pair, ks)
    print(f"l2norm fused : {s2*1e3:.1f} us/call (floor {c2:.2f}ms, R2 {r22:.4f})")
    print(f"  speedup: {s/s2:.2f}x")

    def eager_onorm():
        return onorm_eager(x.reshape(-1, D), g.reshape(-1, D), weight)

    def fused_onorm():
        return onorm_fused(x.reshape(-1, D).astype(mx.float32), g.reshape(-1, D).astype(mx.float32), weight)

    c3, s3, r23, _ = slope_fit(eager_onorm, ks)
    print(f"o_norm eager : {s3*1e3:.1f} us/call (floor {c3:.2f}ms, R2 {r23:.4f})")
    c4, s4, r24, _ = slope_fit(fused_onorm, ks)
    print(f"o_norm fused : {s4*1e3:.1f} us/call (floor {c4:.2f}ms, R2 {r24:.4f})")
    print(f"  speedup: {s3/s4:.2f}x")

    # per-chunk projection: 34 layers x (1 l2norm-pair + 1 onorm)
    e_chunk = (s + s3) * 34
    f_chunk = (s2 + s4) * 34
    print(f"\nper-2048-chunk projection (34 layers): eager {e_chunk:.1f}ms -> fused {f_chunk:.1f}ms (saves {e_chunk - f_chunk:.1f}ms = {(e_chunk-f_chunk)/1876*100:.1f}% of chunk GPU time)")
    print("E-033 KERNEL-EQUIV-PASS")


if __name__ == "__main__":
    main()