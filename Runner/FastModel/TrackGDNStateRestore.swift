// TrackGDNStateRestore.swift -- snapshot/restore for GDN recurrent state.
//
// SEAM. The draft/verify/accept loop lives in vendor EngineLoopV2
// (MTPTargetVerification + MTPFinalize). On reject it calls
// `CBv2RecurrentStateEvaluation.commit(keepPositions:)` or `rollback()`.
// Runner cannot intercept that. The hook we own is the captured verify
// forward: copy each layer's post-position conv/SSM into preallocated
// slots, then `stagePrefixReplay` whose replay closure restores the
// accepted-boundary slot. Restore never re-runs the recurrence.
//
// Default restore is a buffer-view swap plus a device uint32 adopted-slot
// flag: the live state becomes the bank slot, the evicted live view becomes
// that bank entry, and GDN kernels read the flag instead of copying slot
// bytes. `TRACK_ROLLBACK_FLAG=0` keeps w26's copy restore byte-for-byte.
// `TRACK_GDN_STATE_RESTORE=0` keeps the existing `stageCaptured` views.
// The runner manifest leaves `supportsCompactRecurrentMTPReplay` false so
// the advertised digest does not move; admission still charges a captured
// window. Width-1 verify stays on `stageCaptured` (prefix replay needs 2+).

import Foundation
import MLX
import MLXLMCommon

public enum TrackGDNStateRestore {
    /// `TRACK_GDN_STATE_RESTORE=0` falls back to `stageCaptured` views.
    public static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_GDN_STATE_RESTORE"] ?? "1") != "0"
    }()

    /// Default ON. `TRACK_ROLLBACK_FLAG=0` keeps w26's copy restore.
    public static let flagEnabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_ROLLBACK_FLAG"] ?? "1") != "0"
    }()

    /// Seed plus the largest served draft depth (1...6).
    public static let maxVerifyWidth = 1 + TrackQwen4ExpInlineMTPAssistant.maximumDepth

    static func usesPrefixReplay(positions: Int) -> Bool {
        enabled && positions >= 2
    }
}

/// One layer's conv + SSM at one token boundary. CPU mock and aliasing tests.
public struct TrackGDNMockState: Equatable, Sendable {
    public var conv: [Float]
    public var ssm: [Float]

    public init(conv: [Float], ssm: [Float]) {
        self.conv = conv
        self.ssm = ssm
    }
}

/// Preallocated CPU snapshot slots. Flag path: one extra live buffer per
/// layer; restore swaps live with the adopted slot and writes a uint32 flag.
/// Copy path (`useFlag: false`): restore returns a copy, as in w26.
public final class TrackGDNSnapshotBank: @unchecked Sendable {
    public let layerCount: Int
    public let maxPositions: Int
    public let convCount: Int
    public let ssmCount: Int
    public let useFlag: Bool

    private final class SlotStorage {
        var conv: [Float]
        var ssm: [Float]
        let id: UInt32

        init(id: UInt32, convCount: Int, ssmCount: Int) {
            self.id = id
            self.conv = Array(repeating: 0, count: convCount)
            self.ssm = Array(repeating: 0, count: ssmCount)
        }
    }

    private struct LayerPool {
        var slots: [SlotStorage]
        var liveIndex: Int
        var bankIndex: [Int]
        var adoptedFrom: Int?
    }

    private var convSlots: [[Float]]
    private var ssmSlots: [[Float]]
    private var pools: [LayerPool]

    public init(
        layerCount: Int, maxPositions: Int, convCount: Int, ssmCount: Int,
        useFlag: Bool = TrackGDNStateRestore.flagEnabled
    ) {
        precondition(layerCount > 0 && maxPositions > 0 && convCount > 0 && ssmCount > 0)
        self.layerCount = layerCount
        self.maxPositions = maxPositions
        self.convCount = convCount
        self.ssmCount = ssmCount
        self.useFlag = useFlag
        if useFlag {
            self.convSlots = []
            self.ssmSlots = []
            var pools: [LayerPool] = []
            pools.reserveCapacity(layerCount)
            for layer in 0 ..< layerCount {
                let base = UInt32(layer * (maxPositions + 1))
                var slots: [SlotStorage] = []
                slots.reserveCapacity(maxPositions + 1)
                for i in 0 ... maxPositions {
                    slots.append(
                        SlotStorage(
                            id: base + UInt32(i), convCount: convCount, ssmCount: ssmCount))
                }
                pools.append(
                    LayerPool(
                        slots: slots, liveIndex: 0,
                        bankIndex: Array(1 ... maxPositions), adoptedFrom: nil))
            }
            self.pools = pools
        } else {
            let slots = layerCount * maxPositions
            self.convSlots = Array(
                repeating: Array(repeating: 0, count: convCount), count: slots)
            self.ssmSlots = Array(
                repeating: Array(repeating: 0, count: ssmCount), count: slots)
            self.pools = []
        }
    }

    public var allocatedSlotCount: Int {
        useFlag ? layerCount * (maxPositions + 1) : convSlots.count
    }

    public func snapshot(layer: Int, position: Int, state: TrackGDNMockState) {
        precondition(state.conv.count == convCount && state.ssm.count == ssmCount)
        if useFlag {
            precondition(
                layer >= 0 && layer < layerCount && position >= 0 && position < maxPositions)
            var pool = pools[layer]
            let phys = pool.bankIndex[position]
            pool.slots[phys].conv = state.conv
            pool.slots[phys].ssm = state.ssm
            pool.adoptedFrom = nil
            pools[layer] = pool
        } else {
            let index = slotIndex(layer: layer, position: position)
            convSlots[index] = state.conv
            ssmSlots[index] = state.ssm
        }
    }

    public func restore(layer: Int, position: Int) -> TrackGDNMockState {
        if useFlag {
            precondition(
                layer >= 0 && layer < layerCount && position >= 0 && position < maxPositions)
            var pool = pools[layer]
            if pool.adoptedFrom != position {
                var live = pool.liveIndex
                swap(&live, &pool.bankIndex[position])
                pool.liveIndex = live
                pool.adoptedFrom = position
                pools[layer] = pool
            }
            let live = pools[layer].slots[pools[layer].liveIndex]
            return TrackGDNMockState(conv: live.conv, ssm: live.ssm)
        }
        let index = slotIndex(layer: layer, position: position)
        return TrackGDNMockState(conv: convSlots[index], ssm: ssmSlots[index])
    }

    public func adoptedSlot(layer: Int) -> UInt32 {
        liveSlotId(layer: layer)
    }

    public func liveSlotId(layer: Int) -> UInt32 {
        precondition(layer >= 0 && layer < layerCount)
        if useFlag {
            let pool = pools[layer]
            return pool.slots[pool.liveIndex].id
        }
        return UInt32(slotIndex(layer: layer, position: 0))
    }

    public func bankSlotId(layer: Int, position: Int) -> UInt32 {
        if useFlag {
            precondition(
                layer >= 0 && layer < layerCount && position >= 0 && position < maxPositions)
            let pool = pools[layer]
            return pool.slots[pool.bankIndex[position]].id
        }
        return UInt32(slotIndex(layer: layer, position: position))
    }

    private func slotIndex(layer: Int, position: Int) -> Int {
        precondition(
            layer >= 0 && layer < layerCount && position >= 0 && position < maxPositions,
            "TrackGDNSnapshotBank: layer \(layer) position \(position) out of range")
        return layer * maxPositions + position
    }
}

/// Device-side snapshot stacks for the fast-path captured verify window.
/// Flag path: `[maxWidth + 1, ...]` conv/SSM bank per layer, a free-list of
/// slot indices, and one uint32 adoption word. Restore swaps the live index
/// with the adopted bank entry and writes the flag; it does not copy slot
/// bytes. Copy path: `restore` copies a slot, as in w26.
final class TrackGDNStateStore {
    private struct LayerBuffers {
        var conv: MLXArray
        var ssm: MLXArray
        var convBytes: Int
        var ssmBytes: Int
        var liveConv: MLXArray?
        var liveSsm: MLXArray?
    }

    private let maxPositions: Int
    private let useFlag: Bool
    private var layers: [Int: LayerBuffers] = [:]
    private var liveIndex: UInt32 = 0
    private var bankIndex: [UInt32] = []
    private var adoptedFrom: Int?
    private var didAdopt = false
    private var flag: MLXArray

    init(
        maxPositions: Int = TrackGDNStateRestore.maxVerifyWidth,
        useFlag: Bool = TrackGDNStateRestore.flagEnabled
    ) {
        self.maxPositions = maxPositions
        self.useFlag = useFlag
        let word = MLXArray([UInt32(0)])
        eval(word)
        self.flag = word
    }

    var adoptedSlot: UInt32 { liveIndex }

    var deviceFlag: MLXArray { flag }

    /// Copy a captured `[S, ...]` stack into the layer's preallocated slots.
    func capture(modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray) {
        let positions = conv.dim(0)
        precondition(
            positions >= 1 && positions <= maxPositions && ssm.dim(0) == positions,
            "TrackGDNStateStore: captured stacks must be [S, ...] with S in 1...\(maxPositions)")
        let buffers = buffer(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
        if useFlag {
            ensureFreeList()
            for position in 0 ..< positions {
                let row = Int(bankIndex[position])
                let start = MLXArray([Int32(row)])
                let convRow = contiguous(conv[position ..< position + 1])
                let ssmRow = contiguous(ssm[position ..< position + 1])
                buffers.conv._updateInternal(
                    dynamicSliceUpdate(buffers.conv, update: convRow, start: start, axes: [0]))
                buffers.ssm._updateInternal(
                    dynamicSliceUpdate(buffers.ssm, update: ssmRow, start: start, axes: [0]))
            }
            adoptedFrom = nil
        } else {
            let start = MLXArray([Int32(0)])
            buffers.conv._updateInternal(
                dynamicSliceUpdate(buffers.conv, update: contiguous(conv), start: start, axes: [0]))
            buffers.ssm._updateInternal(
                dynamicSliceUpdate(buffers.ssm, update: contiguous(ssm), start: start, axes: [0]))
        }
        layers[modelLayerIndex] = buffers
    }

    /// Stage compact prefix replay. Flag path restores by view-swap + flag;
    /// copy path restores by copy.
    func stagePrefixReplay(
        _ evaluation: CBv2RecurrentStateEvaluation,
        modelLayerIndex: Int,
        positions: Int
    ) throws {
        guard let buffers = layers[modelLayerIndex] else {
            preconditionFailure(
                "TrackGDNStateStore: capture before stage at layer \(modelLayerIndex)")
        }
        let convBytes = buffers.convBytes
        let ssmBytes = buffers.ssmBytes
        let positionBytes = convBytes + ssmBytes
        let last = positions - 1
        let lastPhys = useFlag ? Int(bankIndex[last]) : last
        let roots: [MLXArray] =
            useFlag ? [buffers.conv, buffers.ssm, flag] : [buffers.conv, buffers.ssm]
        // Accept-all next-step start: host-side copy of the last slot, not
        // eval'd. Full accept materializes it at commit; partial drops it.
        // Flag restore uses view-swap instead of this copy.
        let acceptAllCopy: CBv2RecurrentLayerState?
        if !useFlag, TrackStepPrebuild.shouldPrebuild(positions: positions) {
            acceptAllCopy = Self.copySlotUnmaterialized(
                conv: buffers.conv, ssm: buffers.ssm, position: last)
        } else {
            acceptAllCopy = nil
        }
        try evaluation.stagePrefixReplay(
            modelLayerIndex: modelLayerIndex,
            positions: positions,
            finalConv: buffers.conv[lastPhys ..< lastPhys + 1],
            finalSSM: buffers.ssm[lastPhys ..< lastPhys + 1],
            materializedByteCount: positionBytes * positions,
            evaluationRoots: roots,
            strictReplayRetainedByteCount: 0,
            strictReplayRetainedRoots: [],
            fullAcceptanceRetainedByteCount: 0,
            fullAcceptanceRetainedRoots: [],
            fullAcceptance: { [self] in
                if useFlag {
                    return self.adopt(modelLayerIndex: modelLayerIndex, position: last)
                }
                if let acceptAllCopy {
                    eval(acceptAllCopy.conv!, acceptAllCopy.ssm!)
                    return acceptAllCopy
                }
                return Self.copySlot(conv: buffers.conv, ssm: buffers.ssm, position: last)
            },
            replay: { [self] keep in
                if useFlag {
                    return self.adopt(modelLayerIndex: modelLayerIndex, position: keep - 1)
                }
                _ = acceptAllCopy
                return Self.copySlot(conv: buffers.conv, ssm: buffers.ssm, position: keep - 1)
            })
    }

    /// Lazy distinct copy of one snapshot slot. Not eval'd: host graph only.
    func detachSlot(modelLayerIndex: Int, position: Int) -> CBv2RecurrentLayerState? {
        guard let buffers = layers[modelLayerIndex] else { return nil }
        return Self.copySlotUnmaterialized(
            conv: buffers.conv, ssm: buffers.ssm, position: position)
    }

    /// Restore the slot at `position` as live. Flag path swaps views and
    /// writes the adoption word. Copy path copies the slot.
    func adopt(modelLayerIndex: Int, position: Int) -> CBv2RecurrentLayerState {
        guard var buffers = layers[modelLayerIndex] else {
            preconditionFailure(
                "TrackGDNStateStore: capture before adopt at layer \(modelLayerIndex)")
        }
        if !useFlag {
            return Self.copySlot(conv: buffers.conv, ssm: buffers.ssm, position: position)
        }
        ensureFreeList()
        if !(didAdopt && adoptedFrom == position) {
            swap(&liveIndex, &bankIndex[position])
            adoptedFrom = position
            didAdopt = true
            let word = MLXArray([UInt32(liveIndex)])
            flag._updateInternal(word)
            eval(flag)
        }
        let row = Int(liveIndex)
        let liveConv = buffers.conv[row ..< row + 1]
        let liveSsm = buffers.ssm[row ..< row + 1]
        buffers.liveConv = liveConv
        buffers.liveSsm = liveSsm
        layers[modelLayerIndex] = buffers
        return CBv2RecurrentLayerState(conv: liveConv, ssm: liveSsm)
    }

    /// Bank + flag for kernels when `inputConv` is this store's live view.
    func adoptionInputs(modelLayerIndex: Int, inputConv: MLXArray?) -> (
        convBank: MLXArray, ssmBank: MLXArray, flag: MLXArray
    )? {
        guard useFlag, didAdopt, let buffers = layers[modelLayerIndex],
            let inputConv, let live = buffers.liveConv, inputConv === live
        else { return nil }
        return (buffers.conv, buffers.ssm, flag)
    }

    private func ensureFreeList() {
        if bankIndex.isEmpty {
            liveIndex = 0
            bankIndex = (1 ... maxPositions).map { UInt32($0) }
        }
    }

    private func buffer(
        modelLayerIndex: Int, conv: MLXArray, ssm: MLXArray
    ) -> LayerBuffers {
        let convTail = Array(conv.shape.dropFirst())
        let ssmTail = Array(ssm.shape.dropFirst())
        let rows = useFlag ? maxPositions + 1 : maxPositions
        if let existing = layers[modelLayerIndex],
            existing.conv.shape == [rows] + convTail,
            existing.ssm.shape == [rows] + ssmTail,
            existing.conv.dtype == conv.dtype,
            existing.ssm.dtype == ssm.dtype
        {
            return existing
        }
        let convBuf = MLXArray.zeros([rows] + convTail, dtype: conv.dtype)
        let ssmBuf = MLXArray.zeros([rows] + ssmTail, dtype: ssm.dtype)
        eval(convBuf, ssmBuf)
        let allocated = LayerBuffers(
            conv: convBuf, ssm: ssmBuf,
            convBytes: byteCount(shape: convTail, dtype: conv.dtype),
            ssmBytes: byteCount(shape: ssmTail, dtype: ssm.dtype),
            liveConv: nil, liveSsm: nil)
        layers[modelLayerIndex] = allocated
        return allocated
    }

    /// Distinct storage: a later in-place capture cannot alias this slot.
    private static func copySlot(
        conv: MLXArray, ssm: MLXArray, position: Int
    ) -> CBv2RecurrentLayerState {
        let state = copySlotUnmaterialized(conv: conv, ssm: ssm, position: position)
        eval(state.conv!, state.ssm!)
        return state
    }

    /// Host-only copy node. Caller evals on the full-accept submit path;
    /// the discard path drops the node without enqueue.
    fileprivate static func copySlotUnmaterialized(
        conv: MLXArray, ssm: MLXArray, position: Int
    ) -> CBv2RecurrentLayerState {
        let convView = conv[position ..< position + 1]
        let ssmView = ssm[position ..< position + 1]
        let convCopy = convView + MLXArray.zeros(convView.shape, dtype: convView.dtype)
        let ssmCopy = ssmView + MLXArray.zeros(ssmView.shape, dtype: ssmView.dtype)
        return CBv2RecurrentLayerState(conv: convCopy, ssm: ssmCopy)
    }

    private func byteCount(shape: [Int], dtype: DType) -> Int {
        let elements = shape.reduce(1, *)
        return elements * dtype.size
    }
}
