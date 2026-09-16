import Foundation
import MLX

// Preserve load_vector's BF16 subexpressions verbatim, compute
// each lane's transformed activation and bias sum once, and share the float
// packet across this forward's expert/output rows. No cross-forward caching.
enum TrackFastGateUpPacket {
    nonisolated(unsafe) static var enabled = true
    static let packHeader = #"""
        template <typename T>
        inline void track_pack_activation(const device T* x, device float* packet, int j) {
          float values[16];
          const float sum = load_vector<T, float, 16, 4>(x + j * 16, values);
          #pragma clang loop unroll(full)
          for (int v = 0; v < 16; ++v) packet[j * 16 + v] = values[v];
          packet[2560 + j] = sum;
        }
        """#
    static let routePrefix = #"""
        static_assert(VPT == 1 && HAS_GATE && KD == 2560, "serial routed packet");
        const uint prep_sg = simdgroup_index_in_threadgroup;
        if (prep_sg >= 2) {
          for (int j = int((prep_sg - 2) * 32 + thread_index_in_simdgroup);
               j < 160; j += PREP_GROUPS * 32) {
            track_pack_activation(x, packet, j);
          }
          return;
        }
        """#
    static let routeKernel = MLXFast.metalKernel(name: "track_route_activation_packet",
        inputNames: ["logits", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate", "packet"],
        source: routePrefix + "\n" + TrackFastMoEKernels.routeSource,
        header: TrackFastKernels.mixerHeadHeader + TrackFastMoEKernels.wideDecls + packHeader,
        ensureRowContiguous: true)
    static func route(logits: MLXArray, x: MLXArray, gate: TrackQuantWeight, groups: Int) -> [MLXArray] {
        routeKernel([logits, x, gate.weight, gate.scales, gate.biases!],
            template: [("E", 512), ("K", 10), ("T", x.dtype), ("GS", 32), ("BITS", 4),
                       ("KD", 2560), ("VPT", 1), ("HAS_GATE", true), ("PREP_GROUPS", groups)],
            grid: (32, groups + 2, 1), threadGroup: (32, groups + 2, 1),
            outputShapes: [[1, 10], [1, 10], [1], [1, 2720]],
            outputDTypes: [.uint32, .float32, .bfloat16, .float32])
    }
    static let packedHelper: String = {
        var text = TrackFastMoEKernels.gateUpReuseHelpers
        let changes = [
            ("const device T* x,", "const device float* x,"),
            ("x += simd_lid * values_per_thread;", """
                      const device float* sums = x + in_vec_size;
                      x += simd_lid * values_per_thread;
            """),
            ("U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);", """
            U sum = sums[k / values_per_thread + simd_lid];
            #pragma clang loop unroll(full)
            for (int v = 0; v < values_per_thread; ++v) x_thread[v] = x[v];
            """)
        ]
        for (old, new) in changes {
            precondition(text.components(separatedBy: old).count == 2, "packet helper anchor")
            text = text.replacingOccurrences(of: old, with: new)
        }
        return text
    }()
    static let source = TrackFastMoEKernels.gateUpReuseSource.replacingOccurrences(
        of: "x + (size_t)r * (size_t)KD", with: "x + (size_t)r * (size_t)(KD + KD / 16)")
    static let gateKernel = MLXFast.metalKernel(name: "track_gate_up_activation_packet",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
        outputNames: ["act"], source: source,
        header: TrackFastMoEKernels.helpersCore + TrackFastKernels.exactHeader + packedHelper,
        ensureRowContiguous: true)
    static func apply(g: (w: MLXArray, s: MLXArray, b: MLXArray),
                      u: (w: MLXArray, s: MLXArray, b: MLXArray), shared: TrackQuantWeight,
                      packet: MLXArray, ids: MLXArray, xrow: MLXArray) -> MLXArray {
        gateKernel([g.w, g.s, g.b, u.w, u.s, u.b,
                    shared.weight, shared.scales, shared.biases!, packet, ids, xrow],
            template: [("T", g.s.dtype), ("GS", 32), ("BITS", 4), ("N", 640),
                       ("KD", 2560), ("BR", 10), ("RPS", 2)],
            grid: (32, 320, 11), threadGroup: (32, 2, 1),
            outputShapes: [[11, 640]], outputDTypes: [.bfloat16])[0]
    }
    static func moe(_ m: TrackMoE, x: MLXArray, logits: MLXArray,
                    sharedGU: TrackQuantWeight, sharedDown: TrackQuantWeight,
                    replay: (@Sendable ([MLXArray]) -> [MLXArray])?) -> MLXArray? {
        guard enabled, x.shape == [1, 2560], x.dtype == .bfloat16,
            logits.shape == [1, 512], logits.dtype == .float32, m.topK == 10,
            m.expertGroupSize == 32, m.expertBits == 4,
            m.expertGate.w.shape == [512, 640, 320], m.expertUp.w.shape == [512, 640, 320],
            m.expertGate.s.dtype == x.dtype, m.expertUp.s.dtype == x.dtype,
            sharedGU.rows == 1280, sharedGU.groupSize == 32, sharedGU.bits == 4,
            sharedGU.mode == .affine, sharedGU.scales.dtype == x.dtype,
            case .quant(let gate) = m.sharedGate, gate.rows == 1,
            gate.groupSize == 32, gate.bits == 4, gate.mode == .affine,
            gate.biases != nil, gate.scales.dtype == x.dtype else { return nil }
        let r = route(logits: logits, x: x, gate: gate, groups: 5)
        let idx = r[0].reshaped(10)
        let xrow = TrackQwen4ExpFastModel.xrowTable(S: 1, K: 10)
        TrackQwen4ExpFastModel.debugTaps?.append(("moe.activation_packet", x))
        if let replay, StreamOrDevice.default.stream === Stream.gpu {
            return replay([
                r[3], idx, r[1].reshaped(10), r[2], xrow,
                m.expertGate.w, m.expertGate.s, m.expertGate.b,
                m.expertUp.w, m.expertUp.s, m.expertUp.b,
                sharedGU.weight, sharedGU.scales, sharedGU.biases!,
                m.expertDown.w, m.expertDown.s, m.expertDown.b,
                sharedDown.weight, sharedDown.scales, sharedDown.biases!,
            ])[0]
        }
        let act = apply(g: m.expertGate, u: m.expertUp, shared: sharedGU,
                        packet: r[3], ids: idx,
                        xrow: xrow)
        return TrackFastMoEKernels.downCombine(
            wd: m.expertDown.w, sd: m.expertDown.s, bd: m.expertDown.b,
            sharedDown: sharedDown, act: act, idx: idx, w: r[1].reshaped(10),
            gate: r[2], topK: 10, groupSize: m.expertGroupSize, bits: m.expertBits)
    }

}
