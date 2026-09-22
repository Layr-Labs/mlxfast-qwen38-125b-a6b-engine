import Foundation
import MLX

/// A 32-position block recurrence. State storage remains FP32; NAX operands
/// use three BF16 residual components, not a single rounded state tensor.
/// Changes floating-point association; the official token gate is authoritative.
enum TrackGDNChunkBF3 {
    private static let enabled =
        ProcessInfo.processInfo.environment["TRACK_GDN_CHUNK_BF3"] != "0"
    private static let supported: Bool = {
        guard #available(macOS 26.2, iOS 26.2, tvOS 26.2, visionOS 26.2, *) else { return false }
        let arch = GPU.deviceInfo().architecture
        guard let generation = Int(arch.dropLast().suffix(2)), let family = arch.last else { return false }
        return generation >= (family == "p" ? 18 : 17)
    }()

    static func apply(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        state: MLXArray, count: Int, capture: Bool, geometry: TrackFastKernels.GDNGeometry
    ) -> [MLXArray]? {
        guard enabled, supported, !capture, count >= 32, count % 32 == 0,
            StreamOrDevice.default.stream == Stream.gpu,
            geometry.dk == 128, geometry.dv == 128, geometry.hk > 0,
            geometry.hv % geometry.hk == 0,
            q.dtype == .bfloat16, k.dtype == .bfloat16, v.dtype == .bfloat16,
            g.dtype == .float32, beta.dtype == .float32, state.dtype == .float32,
            q.ndim == 4, q.shape == k.shape,
            q.dim(1) == count, q.dim(2) == geometry.hk, q.dim(3) == 128,
            v.shape == [q.dim(0), count, geometry.hv, 128],
            state.shape == [q.dim(0), geometry.hv, 128, 128],
            g.size == q.dim(0) * count * geometry.hv, beta.size == g.size
        else { return nil }
        let B = q.dim(0), chunks = count / 32
        let grams = gramKernel(
            [q, k], template: [("InT", q.dtype), ("T", count), ("HK", geometry.hk), ("CHUNKS", chunks)],
            grid: (32, 4, B * geometry.hk * chunks), threadGroup: (32, 4, 1),
            outputShapes: [[B, geometry.hk, chunks, 32, 32], [B, geometry.hk, chunks, 32, 32]],
            outputDTypes: [.float32, .float32])
        return recurrenceKernel(
            [q, k, v, g, beta, state, grams[0], grams[1]],
            template: [("InT", q.dtype), ("T", count), ("HK", geometry.hk),
                       ("HV", geometry.hv), ("CHUNKS", chunks)],
            grid: (32, 8, B * geometry.hv), threadGroup: (32, 8, 1),
            outputShapes: [[B, count, geometry.hv, 128], state.shape],
            outputDTypes: [.bfloat16, .float32])
    }

    private static let gramKernel = MLXFast.metalKernel(
        name: "track_gdn_chunk_bf3_gram", inputNames: ["q", "k"],
        outputNames: ["kk", "qk"], source: gramSource,
        header: TrackPrefillIndirect.metalHeader, ensureRowContiguous: true)
    private static let recurrenceKernel = MLXFast.metalKernel(
        name: "track_gdn_chunk_bf3_recurrence",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "kk", "qk"],
        outputNames: ["y", "state_out"], source: recurrenceSource,
        header: TrackPrefillIndirect.metalHeader + helpers, ensureRowContiguous: true)

    static let helpers = #"""
    template <typename T>
    METAL_FUNC T gdn_bf3_part(float x, int part) {
      #pragma clang fp reassociate(off)
      #pragma clang fp contract(off)
      const T hi = static_cast<T>(x);
      if (part == 0) { return hi; }
      const float r = x - float(hi);
      const T mid = static_cast<T>(r);
      if (part == 1) { return mid; }
      return static_cast<T>(r - float(mid));
    }

    // S[16,128] times K/Q[32,128]^T. The FP32 state is never overwritten
    // by its BF16 components. Small components accumulate before larger ones.
    template <typename T>
    METAL_FUNC void gdn_bf3_project(
        const thread NAXTile<float, 1, 8>& state,
        const threadgroup T* ks, const threadgroup T* qs,
        thread NAXTile<float, 1, 2>& z, thread NAXTile<float, 1, 2>& out) {
      constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
          16, 32, 16, false, true, true,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
      auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
      auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
      using AT = metal::remove_addrspace_t<decltype(a)>;
      using BT = metal::remove_addrspace_t<decltype(b)>;
      auto cz = op.template get_destination_cooperative_tensor<AT, BT, float>();
      auto co = op.template get_destination_cooperative_tensor<AT, BT, float>();
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 16; ++e) { cz[e] = 0.0f; co[e] = 0.0f; }
      STEEL_PRAGMA_UNROLL
      for (int part = 2; part >= 0; --part) {
        STEEL_PRAGMA_UNROLL
        for (short step = 0; step < 8; ++step) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) { a[e] = gdn_bf3_part<T>(state.val_frags[step][e], part); }
          NAXTile<T, 2, 1> kb, qb;
          kb.template loadV<128>(ks + step * 16);
          qb.template loadV<128>(qs + step * 16);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) {
            b[e] = kb.val_frags[0][e]; b[8 + e] = kb.val_frags[1][e];
          }
          op.run(a, b, cz);
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) {
            b[e] = qb.val_frags[0][e]; b[8 + e] = qb.val_frags[1][e];
          }
          op.run(a, b, co);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          z.val_frags[f][e] = cz[f * 8 + e];
          out.val_frags[f][e] = co[f * 8 + e];
        }
      }
    }

    // S <- decay*S + weighted_innovations[16,32] * K[32,128].
    // kt is the transposed BF16 key stage [128,32].
    template <typename T>
    METAL_FUNC void gdn_bf3_update(
        thread NAXTile<float, 1, 8>& state,
        const thread NAXTile<float, 1, 2>& u,
        const threadgroup T* kt, float decay) {
      constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
          16, 32, 16, false, true, true,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
      auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
      auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
      using AT = metal::remove_addrspace_t<decltype(a)>;
      using BT = metal::remove_addrspace_t<decltype(b)>;
      STEEL_PRAGMA_UNROLL
      for (short block = 0; block < 4; ++block) {
        auto c = op.template get_destination_cooperative_tensor<AT, BT, float>();
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) { c[f * 8 + e] = decay * state.val_frags[2 * block + f][e]; }
        }
        STEEL_PRAGMA_UNROLL
        for (int part = 2; part >= 0; --part) {
          STEEL_PRAGMA_UNROLL
          for (short step = 0; step < 2; ++step) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) { a[e] = gdn_bf3_part<T>(u.val_frags[step][e], part); }
            NAXTile<T, 2, 1> kb;
            kb.template loadV<32>(kt + block * 32 * 32 + step * 16);
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
              b[e] = kb.val_frags[0][e]; b[8 + e] = kb.val_frags[1][e];
            }
            op.run(a, b, c);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) { state.val_frags[2 * block + f][e] = c[f * 8 + e]; }
        }
      }
    }
    """#

    static let gramSource = #"""
      const uint sg = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const uint thread_id = sg * 32 + lane;
      const uint item = threadgroup_position_in_grid.z;
      const uint chunk = item % CHUNKS;
      const uint head = (item / CHUNKS) % HK;
      const uint batch = item / (CHUNKS * HK);
      alignas(16) threadgroup InT ks[32 * 128];
      alignas(16) threadgroup InT qs[32 * 128];
      for (uint i = thread_id; i < 32 * 128; i += 128) {
        const uint t = i / 128, d = i % 128;
        const size_t offset = ((size_t(batch) * T + chunk * 32 + t) * HK + head) * 128 + d;
        ks[i] = k[offset]; qs[i] = q[offset];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
          16, 32, 16, false, true, true,
          mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
      mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
      auto a = op.template get_left_input_cooperative_tensor<InT, InT, float>();
      auto b = op.template get_right_input_cooperative_tensor<InT, InT, float>();
      using AT = metal::remove_addrspace_t<decltype(a)>;
      using BT = metal::remove_addrspace_t<decltype(b)>;
      auto c = op.template get_destination_cooperative_tensor<AT, BT, float>();
      STEEL_PRAGMA_UNROLL
      for (short e = 0; e < 16; ++e) { c[e] = 0.0f; }
      const threadgroup InT* lhs = (sg < 2 ? ks : qs) + (sg % 2) * 16 * 128;
      STEEL_PRAGMA_UNROLL
      for (short step = 0; step < 8; ++step) {
        NAXTile<InT, 1, 1> aa;
        NAXTile<InT, 2, 1> bb;
        aa.template loadV<128>(lhs + step * 16);
        bb.template loadV<128>(ks + step * 16);
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          a[e] = aa.val_frags[0][e]; b[e] = bb.val_frags[0][e]; b[e + 8] = bb.val_frags[1][e];
        }
        op.run(a, b, c);
      }
      device float* destination = (sg < 2 ? kk : qk) + size_t(item) * 32 * 32;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 2; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          const short2 coord = BaseNAXFrag::get_coord(e);
          destination[((sg % 2) * 16 + coord.y) * 32 + f * 16 + coord.x] = c[f * 8 + e];
        }
      }
    """#

    static let recurrenceSource = #"""
      const uint sg = simdgroup_index_in_threadgroup;
      const uint lane = thread_index_in_simdgroup;
      const uint thread_id = sg * 32 + lane;
      const uint item = threadgroup_position_in_grid.z;
      const uint batch = item / HV, head = item % HV;
      const uint key_head = head / (HV / HK);
      const uint row_base = sg * 16;
      alignas(16) threadgroup InT ks[32 * 128];
      // Reused as transposed keys once both state projections have finished.
      alignas(16) threadgroup InT qs[32 * 128];
      threadgroup float ck[32 * 32], cq[32 * 32];
      threadgroup float decay[32], betas[32], prefix[32], tail[32];
      NAXTile<float, 1, 8> state;
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 8; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          const short2 c = BaseNAXFrag::get_coord(e);
          state.val_frags[f][e] = state_in[(size_t(item) * 128 + row_base + c.y) * 128 + f * 16 + c.x];
        }
      }
      for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
        for (uint i = thread_id; i < 32 * 128; i += 256) {
          const uint t = i / 128, d = i % 128;
          const size_t off = ((size_t(batch) * T + chunk * 32 + t) * HK + key_head) * 128 + d;
          ks[i] = k[off]; qs[i] = q[off];
        }
        if (thread_id < 32) {
          const size_t off = (size_t(batch) * T + chunk * 32 + thread_id) * HV + head;
          decay[thread_id] = g[off]; betas[thread_id] = beta[off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        NAXTile<float, 1, 2> u, output;
        gdn_bf3_project<InT>(state, ks, qs, u, output);
        // All SIMD groups have finished reading the query stage before reuse.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = thread_id; i < 32 * 128; i += 256) {
          const uint d = i / 32, t = i % 32;
          qs[i] = ks[t * 128 + d];
        }
        const size_t gram_base = ((size_t(batch) * HK + key_head) * CHUNKS + chunk) * 32 * 32;
        for (uint i = thread_id; i < 32 * 32; i += 256) {
          const uint t = i / 32, j = i % 32;
          float r = t >= j ? 1.0f : 0.0f;
          for (uint p = j + 1; p <= t; ++p) { r *= decay[p]; }
          ck[i] = t > j ? r * kk[gram_base + i] : 0.0f;
          cq[i] = t >= j ? r * qk[gram_base + i] : 0.0f;
        }
        if (thread_id < 32) {
          float p = 1.0f, r = 1.0f;
          for (uint j = 0; j <= thread_id; ++j) { p *= decay[j]; }
          for (uint j = thread_id + 1; j < 32; ++j) { r *= decay[j]; }
          prefix[thread_id] = p; tail[thread_id] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) {
            const short2 c = BaseNAXFrag::get_coord(e);
            const uint t = f * 16 + c.x;
            const size_t off = ((size_t(batch) * T + chunk * 32 + t) * HV + head) * 128 + row_base + c.y;
            u.val_frags[f][e] = float(v[off]) - prefix[t] * u.val_frags[f][e];
            output.val_frags[f][e] *= prefix[t];
          }
        }
        // Forward substitution in register fragments. Four lanes share each
        // value row; broadcast the pivot to those lanes, then update all later
        // times. There is no cross-SIMD dependency in this solve.
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < 32; ++j) {
          const uint source_lane = (lane & ~9u) | (uint(j) & 8u) | ((uint(j) >> 2) & 1u);
          const float low = simd_shuffle(u.val_frags[j / 16][j % 4], source_lane) * betas[j];
          const float high = simd_shuffle(u.val_frags[j / 16][4 + j % 4], source_lane) * betas[j];
          STEEL_PRAGMA_UNROLL
          for (short f = 0; f < 2; ++f) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < 8; ++e) {
              const short2 c = BaseNAXFrag::get_coord(e);
              const uint t = f * 16 + c.x;
              const float pivot = e < 4 ? low : high;
              if (t > uint(j)) { u.val_frags[f][e] -= ck[t * 32 + j] * pivot; }
              else if (t == uint(j)) { u.val_frags[f][e] = pivot; }
              if (t >= uint(j)) { output.val_frags[f][e] += cq[t * 32 + j] * pivot; }
            }
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short f = 0; f < 2; ++f) {
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < 8; ++e) {
            const short2 c = BaseNAXFrag::get_coord(e);
            const uint t = f * 16 + c.x;
            const size_t off = ((size_t(batch) * T + chunk * 32 + t) * HV + head) * 128 + row_base + c.y;
            y[off] = static_cast<InT>(output.val_frags[f][e]);
            u.val_frags[f][e] *= tail[t];
          }
        }
        gdn_bf3_update<InT>(state, u, qs, prefix[31]);
        // Protect both shared stages and scalar metadata from the next chunk.
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      STEEL_PRAGMA_UNROLL
      for (short f = 0; f < 8; ++f) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          const short2 c = BaseNAXFrag::get_coord(e);
          state_out[(size_t(item) * 128 + row_base + c.y) * 128 + f * 16 + c.x] = state.val_frags[f][e];
        }
      }
    """#
}
