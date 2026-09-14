// TrackKVFuse.swift -- fused RMSNorm + RoPE + in-place KV append for the 12
// full-attention layers (PB-177 / reusable-playbook C10).
//
// SEAM. TrackFastKernels.attnPrep writes q/k/v into fresh arrays; the cache
// then slice-updates those K/V rows into the ring at the known write offset.
// The fused kernel keeps attnPrep's arithmetic (rms_single_row layout, weight
// after the bf16 rounding of x*inv_mean, composed partial rope over bf16
// cos/sin) and also stores rope'd K and copied V at the ring slot, so the
// values are not written to a temp and re-read by the append.
//
// GATE. Full-attn only. Decode and prefill when the ring offset and capacity
// are known at dispatch and the window fits without growing the buffer.
// Otherwise the separate attnPrep + cache.update path. GDN/QSA layers never
// enter this file.
//
// THREADGROUP. Same arrays as attnPrep: float local_sums[32] + InT vec[D].
// slice_update's copy uses no threadgroup memory, so the separate-path
// maximum is attnPrep's 640 B at D=256 bf16. Fused equals that maximum and
// stays under the 32 KB API cap and the ~60 KB Apple7/8 physical pool
// (PB-607 / PB-430).
//
// TRACK_KV_FUSE default ON; `0` restores the separate path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

enum TrackKVFuse {
    /// `TRACK_KV_FUSE=0` restores attnPrep + cache slice-update.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_KV_FUSE"] ?? "1") != "0"
    }()

    /// Production full-attn geometry (docs/participant-contract.md).
    static let productionHeads = 24
    static let productionKVHeads = 2
    static let productionHeadDim = 256
    static let productionRotaryDims = 64
    static let productionLayers = 12
    static let productionElementBytes = 2

    // MARK: - Threadgroup pricing (PB-430 / PB-607)

    struct ThreadgroupPrice: Equatable, Sendable {
        let headDim: Int
        let elementBytes: Int

        static let nReads = 4
        static let localSumsCount = 32
        static let localSumsBytes = 32 * MemoryLayout<Float>.size
        /// 32 KB Metal API threadgroup cap.
        static let apiCapBytes = 32 * 1024
        /// Apple7/8 physical threadgroup pool per core (PB-607).
        static let physicalBytesPerCore = 60 * 1024

        var threadsPerThreadgroup: Int { headDim / Self.nReads }
        var vecBytes: Int { headDim * elementBytes }
        /// attnPrep / fused body: `threadgroup float local_sums[32]` + `threadgroup InT vec[D]`.
        var fusedBytes: Int { Self.localSumsBytes + vecBytes }
        /// slice_update's copy_gg kernel keeps the update in registers.
        var sliceUpdateBytes: Int { 0 }
        var separateMaxBytes: Int { max(fusedBytes, sliceUpdateBytes) }
        var fitsSeparateMax: Bool { fusedBytes <= separateMaxBytes }
        var fitsAPICap: Bool { fusedBytes <= Self.apiCapBytes }
        var residentThreadgroupsPerCore: Int {
            Self.physicalBytesPerCore / max(fusedBytes, 1)
        }

        static let production = ThreadgroupPrice(
            headDim: TrackKVFuse.productionHeadDim,
            elementBytes: TrackKVFuse.productionElementBytes)
    }

    /// K+V bytes attnPrep used to write and the append used to re-read, per step.
    static func bytesRoundTripRemoved(seq: Int, layers: Int = productionLayers) -> Int {
        2 * productionKVHeads * seq * productionHeadDim * productionElementBytes * layers
    }

    // MARK: - Ring index

    /// Slot in a capacity-`cap` ring. `wrap` is the prefill/decode wrap at
    /// the boundary: at most one wrap when `count < cap`.
    static func ringSlot(offset: Int, token: Int, cap: Int, wrap: Bool) -> Int {
        let slot = offset + token
        if wrap, slot >= cap { return slot - cap }
        return slot
    }

    static func canAppend(offset: Int, count: Int, cap: Int, wrap: Bool) -> Bool {
        guard offset >= 0, count >= 0, cap > 0, count <= cap else { return false }
        if wrap { return true }
        return offset + count <= cap
    }

    // MARK: - Live KV buffers (same MLXArray instances the cache holds)

    struct Ring {
        let keys: MLXArray
        let values: MLXArray
        let offset: Int
        var cap: Int { keys.dim(2) }
    }

    static func ring(from cache: Qwen4ExpCBv2LayerCache) -> Ring? {
        guard cache.rows.count == 1 else { return nil }
        let inner = cache.innerState()
        // CBv2LayerCache.innerState: [positionOffsets, keys, values, ...tape]
        guard inner.count >= 3 else { return nil }
        let keys = inner[1], values = inner[2]
        guard keys.ndim == 4, values.ndim == 4,
            keys.dim(0) == 1, values.dim(0) == 1,
            keys.dim(2) == values.dim(2), keys.dim(2) > 0,
            keys.dtype == values.dtype
        else { return nil }
        return Ring(keys: keys, values: values, offset: cache.rows[0].absoluteOffset)
    }

    static func ring(from cache: Qwen4ExpAttentionCache) -> Ring? {
        let inner = cache.innerState()
        guard inner.count >= 2 else { return nil }
        let keys = inner[0], values = inner[1]
        guard keys.ndim == 4, values.ndim == 4,
            keys.dim(2) == values.dim(2), keys.dim(2) > 0,
            keys.dtype == values.dtype
        else { return nil }
        return Ring(keys: keys, values: values, offset: cache.offset)
    }
}
