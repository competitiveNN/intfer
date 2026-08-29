// examples/spinquant_test/main.cpp
// Standalone verification of Stage-1 step 1 (SpinQuant). No SYCL required.
//
// 1. Builds a tiny FP16 transformer-ish ModelIR with deliberate outliers.
// 2. Runs spinquant_preprocess (Hadamard, and Learned when calibration is given).
// 3. Asserts function preservation:  y = x W^T  ==  x' (W')^T  (R orthogonal).
// 4. Measures activation outlier reduction after the online rotation (x' = x R):
//       - max |x| before vs after
//       - per-group INT8 symmetric quantization MSE before vs after
// 5. Round-trips the RotationSet through serialize/deserialize.

#include "intfer/spinquant.hpp"
#include "intfer/quant_spec.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace intfer;

// x R  (right-multiply activation by rotation), x has length dim.
std::vector<float> apply_online_right(const std::vector<float>& x, const Rotation& R) {
    std::vector<float> out(R.dim, 0.0f);
    for (uint32_t j = 0; j < R.dim; ++j) {
        float s = 0.0f;
        for (uint32_t k = 0; k < R.dim; ++k) s += x[k] * R.mat[static_cast<size_t>(j) * R.dim + k];
        out[j] = s;
    }
    return out;
}

// Symmetric per-group INT8 (maxabs) quantization MSE of one vector.
double sym_group_mse(const std::vector<float>& v, int G) {
    if (v.empty()) return 0.0;
    double err = 0.0;
    int ng = static_cast<int>((v.size() + G - 1) / G);
    for (int g = 0; g < ng; ++g) {
        int s = g * G, e = std::min(static_cast<int>(v.size()), s + G);
        float mx = 0.0f;
        for (int i = s; i < e; ++i) mx = std::max(mx, std::fabs(v[i]));
        if (mx == 0.0f) continue;
        float scale = mx / 127.0f;
        for (int i = s; i < e; ++i) {
            float q = std::round(v[i] / scale);
            q = std::max(-127.0f, std::min(127.0f, q));
            float d = v[i] - q * scale;
            err += static_cast<double>(d) * d;
        }
    }
    return err / v.size();
}

float maxabs(const std::vector<float>& v) {
    float m = 0.0f;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

// Build a layer-0 QKV + FFN block with outliers in both weights and activations.
ModelIR build_model(std::mt19937& rng, std::vector<std::vector<float>>& calib_out) {
    ModelIR m;
    m.model_id = "test";
    m.n_layers = 1;

    const uint32_t K = 32;   // contraction dim (power of 2 => exact Hadamard)
    const uint32_t N = 32;   // output dim
    const char* names[5] = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    };
    TensorRole roles[5] = { TensorRole::ATTN_Q, TensorRole::ATTN_K, TensorRole::ATTN_V,
                            TensorRole::MLP_GATE, TensorRole::MLP_DOWN };

    std::normal_distribution<float> nd(0.0f, 0.05f);
    size_t fp16_cursor = 0;
    for (int t = 0; t < 5; ++t) {
        TensorIR ti;
        ti.name = names[t];
        ti.role = roles[t];
        ti.N = N; ti.K = K;
        ti.qcfg.scheme = QuantScheme::FP16;
        ti.fp16_offset = fp16_cursor;
        ti.fp16_bytes = static_cast<uint64_t>(N) * K * 2;
        fp16_cursor += N * K;
        m.tensors.push_back(ti);
    }
    m.fp16_buf.resize(fp16_cursor);

    // Fill weights with small Gaussian + a few large outliers.
    std::vector<uint16_t> buf(fp16_cursor);
    for (size_t i = 0; i < fp16_cursor; ++i) {
        float v = nd(rng);
        if (i % 97 == 0) v += (i % 2 ? 8.0f : -8.0f);  // injected outlier
        buf[i] = f32_to_f16(v);
    }
    for (size_t i = 0; i < fp16_cursor; ++i) m.fp16_buf[i] = buf[i];

    // Calibration activations (input to q/k/v, which share QKV site) with outliers.
    const int n_seq = 4, n_tok = 64;
    std::vector<float> act;
    for (int s = 0; s < n_seq; ++s)
        for (int tk = 0; tk < n_tok; ++tk) {
            for (uint32_t i = 0; i < K; ++i) {
                float v = nd(rng);
                if (i == 3 || i == 17) v += 6.0f;  // strong channel outliers
                act.push_back(v);
            }
        }
    calib_out.push_back(act);
    return m;
}

}  // namespace

int main() {
    std::mt19937 rng(12345);
    std::vector<std::vector<float>> calib_vec;
    ModelIR model = build_model(rng, calib_vec);

    CalibrationData calib;
    calib.activations.emplace_back("model.layers.0.self_attn.q_proj.weight", calib_vec[0]);

    bool ok = true;

    // ---- Hadamard path ----
    SpinQuantConfig cfg;
    cfg.type = RotationType::Hadamard;
    cfg.per_layer = false;
    SpinQuantResult res = spinquant_preprocess(model, cfg, &calib);
    std::printf("[spinquant] Hadamard: %zu rotation(s) generated\n", res.rotations.size());

    // Find the QKV rotation to demonstrate activation outlier reduction + preservation.
    const Rotation* Rq = nullptr;
    for (const auto& r : res.rotations.rotations)
        if (r.site == RotationSite::QKV) { Rq = &r; break; }
    if (!Rq) { std::printf("FAIL: no QKV rotation\n"); return 1; }

    // Outlier reduction on a calibration token (first token of first sequence).
    std::vector<float> x(Rq->dim);
    const float* base = calib_vec[0].data();
    for (uint32_t i = 0; i < Rq->dim; ++i) x[i] = base[i];
    std::vector<float> xr = apply_online_right(x, *Rq);

    float mx_before = maxabs(x), mx_after = maxabs(xr);
    double mse_before = sym_group_mse(x, 64), mse_after = sym_group_mse(xr, 64);
    std::printf("  activation max|x|: %.4f -> %.4f  (x%.2f lower)\n",
                mx_before, mx_after, mx_before / std::max(mx_after, 1e-6f));
    std::printf("  activation qMSE(G64): %.4g -> %.4g  (x%.2f lower)\n",
                mse_before, mse_after, mse_before / std::max(mse_after, 1e-12));

    if (!(mx_after <= mx_before + 1e-3f)) { std::printf("FAIL: outlier not reduced\n"); ok = false; }
    if (!(mse_after < mse_before))        { std::printf("FAIL: quant MSE not reduced by rotation\n"); ok = false; }

    // ---- Function preservation: y = x W^T  vs  y' = x' (W')^T ----
    // Note: spinquant_preprocess copies the input, so `model` still holds the
    // ORIGINAL weights while `res.model` holds the rotated ones.
    {
        const TensorIR& torig = model.tensors[0];        // q_proj, original
        const TensorIR& trot  = res.model.tensors[0];    // q_proj, rotated
        std::vector<float> Worig = load_tensor_fp16(model, torig);
        std::vector<float> Wrot  = load_tensor_fp16(res.model, trot);
        uint32_t N = torig.N, K = torig.K;
        std::vector<float> xin(K);
        for (uint32_t i = 0; i < K; ++i) xin[i] = (i == 3) ? 6.0f : 0.05f * (static_cast<int>(i) - 16);

        // y = x W^T  (original)
        std::vector<float> y(N, 0.0f);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) y[n] += xin[k] * Worig[static_cast<size_t>(n) * K + k];

        // x' = x R ;  W' = W R^T (already folded into trot)
        std::vector<float> xr2 = apply_online_right(xin, *Rq);
        std::vector<float> yp(N, 0.0f);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t k = 0; k < K; ++k) yp[n] += xr2[k] * Wrot[static_cast<size_t>(n) * K + k];

        double max_diff = 0.0;
        for (uint32_t n = 0; n < N; ++n) max_diff = std::max(max_diff, std::fabs(static_cast<double>(y[n]) - static_cast<double>(yp[n])));
        std::printf("  function-preservation max|y - y'|: %.3e\n", max_diff);
        if (!(max_diff < 1e-2)) { std::printf("FAIL: rotation changed the function\n"); ok = false; }
    }

    // ---- Serialize / deserialize round-trip ----
    {
        std::vector<uint8_t> blob = serialize_rotations(res.rotations);
        RotationSet back = deserialize_rotations(blob.data(), blob.size());
        bool same = (back.size() == res.rotations.size());
        for (size_t i = 0; same && i < back.size(); ++i) {
            const Rotation& a = res.rotations.rotations[i];
            const Rotation& b = back.rotations[i];
            if (a.id != b.id || a.dim != b.dim || a.mat.size() != b.mat.size()) same = false;
            else for (size_t j = 0; j < a.mat.size(); ++j)
                if (std::fabs(a.mat[j] - b.mat[j]) > 1e-3f) { same = false; break; }
        }
        std::printf("  serialize round-trip: %s\n", same ? "OK" : "FAIL");
        if (!same) ok = false;
    }

    // ---- Learned path (uses calibration) ----
    {
        SpinQuantConfig cfg2;
        cfg2.type = RotationType::Learned;
        cfg2.per_layer = false;
        cfg2.learn_iters = 12;
        SpinQuantResult rl = spinquant_preprocess(model, cfg2, &calib);
        const Rotation* Rl = nullptr;
        for (const auto& r : rl.rotations.rotations)
            if (r.site == RotationSite::QKV) { Rl = &r; break; }
        std::vector<float> xrl = apply_online_right(x, *Rl);
        double mse_learned = sym_group_mse(xrl, 64);
        std::printf("  learned qMSE(G64): %.4g  (vs hadamard %.4g)\n", mse_learned, mse_after);
        if (!(mse_learned <= mse_after + 1e-9)) {
            std::printf("WARN: learned path did not improve over Hadamard (acceptable)\n");
        }
    }

    std::printf(ok ? "[spinquant] PASS\n" : "[spinquant] FAIL\n");
    return ok ? 0 : 1;
}
