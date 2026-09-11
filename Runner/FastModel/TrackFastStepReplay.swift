// TrackFastStepReplay.swift -- compiled replay of the serial step's
// shape-stable subgraphs, and the placement of the step's first dispatch.
//
// WHAT THIS IS. A one-token decode step builds several hundred dependent
// launches whose shapes never change from step to step, except in two places:
// the attention layers' key/value views grow by one position per step, and the
// n-gram block reads its rows on the host from the token that was just
// sampled. Everything else -- the 36 gated-deltanet layers, every mixer, every
// routed-expert launch -- is the same graph at the same shapes on every step.
//
// `MLX.compile` traces such a subgraph ONCE and replays the recorded tape on
// later calls, so the host rebuilds nothing. The tape is the same sequence of
// the same kernels over the same buffers: no kernel here is replaced and no
// accumulation is re-associated, so the outputs are bit-identical to the
// uncompiled path (`MLXFAST_CR_REPLAY_VERIFY=1` checks that element for
// element against the graph built op by op beside it).
//
// ONE trace serves every layer of a family. The weights ride as inputs rather
// than as captured constants, so the 36 deltanet layers share a single tape
// instead of recording 36 of them.
//
// Toggles, all default ON except the checker:
//   MLXFAST_CR_EARLY_DISPATCH=0   -- first dispatch after the n-gram layer
//   MLXFAST_CR_GDN_REPLAY=0       -- uncompiled deltanet layers
//   MLXFAST_CR_MOE_REPLAY=0       -- uncompiled expert block
//   MLXFAST_CR_REPLAY_VERIFY=1    -- run both paths and compare
//   MLXFAST_CR_LOG=<path>         -- where the checker writes (default stderr)

import Foundation
import MLX
import MLXNN

enum TrackFastStepReplay {

    static func flag(_ name: String, default value: Bool) -> Bool {
        guard let raw = ProcessInfo.processInfo.environment[name] else { return value }
        return raw != "0"
    }

    /// Dispatch the first layer BEFORE the n-gram host gather.
    ///
    /// A one-token step depends on the token the previous step sampled, so
    /// nothing in it can be queued before that token exists. The n-gram block
    /// at layer index 1 then reads its rows ON THE HOST (the table is
    /// SSD-offloaded behind a bounded LRU), and those rows are addressed by
    /// that same token -- so the host blocks there. With the tip's cadence the
    /// step's first `asyncEval` lands after that block, which leaves the GPU
    /// with an empty queue for the whole gather plus the build of two layers.
    /// Dispatching at the end of layer 0 instead hands the GPU a layer of work
    /// first. This is placement only: `asyncEval` moves no arithmetic, and the
    /// graph it evaluates is the same graph.
    static let earlyDispatch: Bool = flag("MLXFAST_CR_EARLY_DISPATCH", default: true)

    /// Replay the whole gated-deltanet body (input projection, prep, the
    /// recurrence, the gated norm and the output projection) as one tape.
    static let gdnEnabled: Bool = flag("MLXFAST_CR_GDN_REPLAY", default: true)

    /// Replay the whole one-token expert block (router GEMV, top-k +
    /// softmax, the shared expert's gate, gate|up and down+combine).
    static let moeEnabled: Bool = flag("MLXFAST_CR_MOE_REPLAY", default: true)

    /// Bit-check mode: run the uncompiled graph beside every replay and
    /// report any element that differs. Diagnostic only -- it doubles the
    /// work of every replayed block.
    static let verify: Bool = flag("MLXFAST_CR_REPLAY_VERIFY", default: false)

    /// The checker's output: stderr, or the file `MLXFAST_CR_LOG` names when a
    /// driver drains the worker's stderr rather than showing it. Reached only
    /// in the diagnostic mode above; a normal run never opens it.
    nonisolated(unsafe) private static let sink: FileHandle = {
        guard let path = ProcessInfo.processInfo.environment["MLXFAST_CR_LOG"],
            !path.isEmpty
        else { return FileHandle.standardError }
        FileManager.default.createFile(atPath: path, contents: nil)
        return (try? FileHandle(forWritingTo: URL(fileURLWithPath: path)))
            ?? FileHandle.standardError
    }()

    static func log(_ message: String) {
        sink.write(Data(message.utf8))
    }

    nonisolated(unsafe) private static var verified: [String: Int] = [:]
    private static let verifyLock = NSLock()
    static let lock = NSLock()

    /// Element-for-element comparison of a replayed tape against the same
    /// graph built op by op.
    static func compare(_ tag: String, _ pairs: [(String, MLXArray, MLXArray)]) {
        for (name, got, want) in pairs {
            precondition(
                got.shape == want.shape && got.dtype == want.dtype,
                "TrackFastStepReplay: \(tag).\(name) shape/dtype drift "
                    + "\(got.shape)/\(got.dtype) vs \(want.shape)/\(want.dtype)")
            let bad = differingElements(got, want)
            verifyLock.lock()
            let seen = verified[tag + "." + name, default: 0]
            verified[tag + "." + name] = seen + 1
            verifyLock.unlock()
            if bad != 0 {
                log("[cr-verify] MISMATCH \(tag).\(name): \(bad) elements\n")
            } else if seen == 0 {
                log("[cr-verify] ok \(tag).\(name) \(got.shape) \(got.dtype)\n")
            }
        }
    }

    /// float32 is a lossless widening of every dtype in play here, so a
    /// last-bit difference in the narrow type survives the comparison.
    private static func differingElements(_ a: MLXArray, _ b: MLXArray) -> Int {
        let diff = (a.asType(.float32).flattened() .!= b.asType(.float32).flattened())
            .asType(.int32).sum()
        eval(diff)
        return diff.item(Int.self)
    }


    // MARK: - the expert block

    nonisolated(unsafe) private static var moeFn: (@Sendable ([MLXArray]) -> [MLXArray])? = nil
    nonisolated(unsafe) private static var moeKey: String? = nil

    /// The one expert-block tape: router GEMV, then the fused top-k + softmax
    /// + shared-gate launch, the routed gate|up SwiGLU and the routed
    /// down+combine. Inputs
    /// `[x, xF32, routerW, gateW/S/B, expGateW/S/B, expUpW/S/B,
    ///   sharedGUW/S/B, expDownW/S/B, sharedDownW/S/B, xrow]`;
    /// output `[combined]`.
    private static func moeBlock(
        key: String, topK: Int, hidden: Int, expertGroup: Int, expertBits: Int,
        gateGroup: Int, gateBits: Int, gateMode: QuantizationMode,
        guGroup: Int, guBits: Int, guMode: QuantizationMode,
        downGroup: Int, downBits: Int, downMode: QuantizationMode
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        lock.lock()
        defer { lock.unlock() }
        if let existing = moeFn { return moeKey == key ? existing : nil }
        let built: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) { inputs in
            let logits = TrackFastMoEKernels.routerGemv(x: inputs[1], w: inputs[2])
                .reshaped(1, 1, -1)
            let x2 = inputs[0].reshaped(1, hidden)
            let gateQ = TrackQuantWeight(
                weight: inputs[3], scales: inputs[4], biases: inputs[5],
                groupSize: gateGroup, bits: gateBits, mode: gateMode)
            let r = TrackFastMoEKernels.route(
                logits: logits.reshaped(1, -1), x: x2, sharedGate: gateQ, topK: topK)
            let flatIdx = r.idx.reshaped(topK)
            let sharedGU = TrackQuantWeight(
                weight: inputs[12], scales: inputs[13], biases: inputs[14],
                groupSize: guGroup, bits: guBits, mode: guMode)
            let act = TrackFastMoEKernels.gateUpAct(
                wg: inputs[6], sg: inputs[7], bg: inputs[8],
                wu: inputs[9], su: inputs[10], bu: inputs[11], shared: sharedGU,
                x: x2, idx: flatIdx, xrow: inputs[21],
                groupSize: expertGroup, bits: expertBits)
            let sharedDown = TrackQuantWeight(
                weight: inputs[18], scales: inputs[19], biases: inputs[20],
                groupSize: downGroup, bits: downBits, mode: downMode)
            return [
                TrackFastMoEKernels.downCombine(
                    wd: inputs[15], sd: inputs[16], bd: inputs[17], sharedDown: sharedDown,
                    act: act, idx: flatIdx, w: r.w.reshaped(topK), gate: r.gate, topK: topK,
                    groupSize: expertGroup, bits: expertBits)
            ]
        }
        moeKey = key
        moeFn = built
        return built
    }

    /// The per-layer operands of the expert-block tape. Declines unless this
    /// layer is the one-token geometry the tape records -- the same admission
    /// the uncompiled one-token path applies: a 4-bit one-row shared-expert
    /// gate the top-k launch carries, quantized shared gate|up and down, and a
    /// router shape the one-token router GEMV serves.
    static func moeOperands(_ m: TrackMoE, hidden: Int) -> (
        fn: @Sendable ([MLXArray]) -> [MLXArray], weights: [MLXArray]
    )? {
        guard moeEnabled,
            case .quant(let gate) = m.sharedGate, let gateB = gate.biases,
            gate.rows == 1, gate.bits == 4,
            case .quant(let gu)? = m.sharedGateUp.fused, let guB = gu.biases,
            case .quant(let down) = m.sharedDown, let downB = down.biases,
            m.routerW16.dtype == .bfloat16,
            hidden % 128 == 0, m.routerW16.dim(0) % 16 == 0, m.routerW16.dim(0) < 4096,
            hidden < 16 * m.routerW16.dim(0), m.topK <= 32
        else { return nil }
        let key = [
            m.topK, hidden, m.expertGroupSize, m.expertBits, gate.groupSize, gate.bits,
            gu.groupSize, gu.bits, down.groupSize, down.bits, m.routerW16.dim(0),
        ].map(String.init).joined(separator: ":")
        guard let fn = moeBlock(
            key: key, topK: m.topK, hidden: hidden,
            expertGroup: m.expertGroupSize, expertBits: m.expertBits,
            gateGroup: gate.groupSize, gateBits: gate.bits, gateMode: gate.mode,
            guGroup: gu.groupSize, guBits: gu.bits, guMode: gu.mode,
            downGroup: down.groupSize, downBits: down.bits, downMode: down.mode)
        else { return nil }
        return (
            fn,
            [
                m.routerW16,
                gate.weight, gate.scales, gateB,
                m.expertGate.w, m.expertGate.s, m.expertGate.b,
                m.expertUp.w, m.expertUp.s, m.expertUp.b,
                gu.weight, gu.scales, guB,
                m.expertDown.w, m.expertDown.s, m.expertDown.b,
                down.weight, down.scales, downB,
            ]
        )
    }

    // MARK: - gated deltanet

    /// Geometry every deltanet layer of this checkpoint shares; the first
    /// bound layer fixes it and the rest are checked against it.
    struct GDNShape: Equatable {
        let geometry: TrackFastKernels.GDNGeometry
        let zOffset: Int
        let projGroup: Int
        let projBits: Int
        let projMode: QuantizationMode
        let outGroup: Int
        let outBits: Int
        let outMode: QuantizationMode
    }

    nonisolated(unsafe) private static var gdnShape: GDNShape? = nil
    nonisolated(unsafe) private static var gdnFn: (@Sendable ([MLXArray]) -> [MLXArray])? = nil

    /// The one deltanet tape. Inputs:
    /// `[x, convState, ssm, projW, projS, projB, convW, negExpALog, dtBias,
    ///   normW, outW, outS, outB]`; outputs `[out, convOut, stateOut]`.
    private static func gdnBody(_ shape: GDNShape) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        lock.lock()
        defer { lock.unlock() }
        if let existing = gdnFn {
            // A second geometry would need a second tape; this tower has one.
            return gdnShape == shape ? existing : nil
        }
        let geometry = shape.geometry
        let zOffset = shape.zOffset
        let built: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) {
            [projGroup = shape.projGroup, projBits = shape.projBits, projMode = shape.projMode,
             outGroup = shape.outGroup, outBits = shape.outBits, outMode = shape.outMode] inputs in
            let projW = TrackQuantWeight(
                weight: inputs[3], scales: inputs[4], biases: inputs[5],
                groupSize: projGroup, bits: projBits, mode: projMode)
            let proj = projW.apply(inputs[0])
            let r = TrackFastKernels.gdn(
                proj: proj, convState: inputs[1], convW: inputs[6], negExpALog: inputs[7],
                dtBias: inputs[8], stateIn: inputs[2], T: 1, capture: false, geometry: geometry)
            let gated = TrackFastKernels.gatedRMS(
                y: r.y, proj: proj, w: inputs[9], zOffset: zOffset, eps: 1e-6)
            let outW = TrackQuantWeight(
                weight: inputs[10], scales: inputs[11], biases: inputs[12],
                groupSize: outGroup, bits: outBits, mode: outMode)
            return [outW.apply(gated), r.convOut, r.stateOut]
        }
        gdnShape = shape
        gdnFn = built
        return built
    }

    /// The per-layer weight operands of the deltanet tape, or nil when this
    /// layer's projections are not the quantized geometry the tape records.
    static func gdnOperands(_ g: TrackGDN) -> (
        fn: @Sendable ([MLXArray]) -> [MLXArray], weights: [MLXArray]
    )? {
        guard gdnEnabled,
            case .quant(let proj)? = g.proj.fused, let projB = proj.biases,
            case .quant(let out) = g.out, let outB = out.biases
        else { return nil }
        let shape = GDNShape(
            geometry: g.geometry, zOffset: g.zOffset,
            projGroup: proj.groupSize, projBits: proj.bits, projMode: proj.mode,
            outGroup: out.groupSize, outBits: out.bits, outMode: out.mode)
        guard let fn = gdnBody(shape) else { return nil }
        return (
            fn,
            [
                proj.weight, proj.scales, projB,
                g.convW, g.negExpALog, g.dtBias, g.normW,
                out.weight, out.scales, outB,
            ]
        )
    }
}
