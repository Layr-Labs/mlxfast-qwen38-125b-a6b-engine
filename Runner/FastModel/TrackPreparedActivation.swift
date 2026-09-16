import Foundation
import MLX

/// Qwen-27B's producer/consumer activation-table idea, adapted to this
/// checkpoint's affine group-32 decode kernel and its exact load_vector tree.
/// One table is shared by all routed and shared gate/up output rows.
extension TrackFastMoEKernels {
    static let prepareActivationEnabled =
        ProcessInfo.processInfo.environment["MLX_TRACK_PREPARED_ACTIVATION"] != "0"

    static let preparedRouteSource = """
        if (simdgroup_index_in_threadgroup == 0) {
            const uint lane0 = thread_index_in_simdgroup;
            for (uint block = 0; block < (uint)KD / 512; ++block) {
                const uint offset = block * 512 + lane0 * 16;
                float values[16];
                const float total = load_vector<T, float, 16, 4>(x + offset, values);
                for (uint i = 0; i < 16; ++i) { prepared[offset + i] = values[i]; }
                prepared[KD + block * 32 + lane0] = total;
            }
        }
        """ + routeSource.replacingOccurrences(
            of: "constexpr uint SEL_SG = (VPT == 1 && HAS_GATE) ? 1u : 0u;",
            with: "constexpr uint SEL_SG = 1u;")

    nonisolated(unsafe) static let preparedRouteKernel = MLXFast.metalKernel(
        name: "track_moe_route_prepare_activation_v1",
        inputNames: ["logits", "x", "wg", "sgw", "bgw"],
        outputNames: ["idx", "w", "gate", "prepared"],
        source: preparedRouteSource,
        header: TrackFastKernels.mixerHeadHeader + wideDecls, ensureRowContiguous: true)

    // Preserve the existing qdot and accumulation tree literally. Only the
    // input-independent-of-weight-row load/scale/sum work moves to the router.
    static let preparedGateHelpers = gateUpReuseHelpers
        .replacingOccurrences(of: "const device T* x,", with: "const device T* x, const device float* prepared,")
        .replacingOccurrences(
            of: "U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);",
            with: """
                U sum = prepared[in_vec_size + (k / block_size) * SIMD_SIZE + simd_lid];
                for (int i = 0; i < values_per_thread; ++i) {
                    x_thread[i] = prepared[k + simd_lid * values_per_thread + i];
                }
                """)

    static let preparedGateSource = gateUpReuseSource.replacingOccurrences(
        of: "KD, out_row, thread_index_in_simdgroup, g, u);",
        with: "prepared, KD, out_row, thread_index_in_simdgroup, g, u);")

    nonisolated(unsafe) static let preparedGateKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_prepared_activation_v1",
        inputNames: ["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow", "prepared"],
        outputNames: ["act"], source: preparedGateSource,
        header: helpersCore + TrackFastKernels.exactHeader + preparedGateHelpers,
        ensureRowContiguous: true)
}
