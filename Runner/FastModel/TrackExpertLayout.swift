import CoreFoundation
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Routed-expert storage layout. Default ON. `TRACK_EXPERT_LAYOUT=0`, a
/// checkpoint without `mlxfast-expert-layout.json`, and an older table
/// version all keep the split `gate_proj` / `up_proj` / `down_proj` modules
/// the vendored loader already serves.
enum TrackExpertLayout {
    static let tableFileName = "mlxfast-expert-layout.json"
    static let currentVersion = 2
    static let currentLayout = "gate_up_down_per_expert"

    static var enabled: Bool {
        (ProcessInfo.processInfo.environment["TRACK_EXPERT_LAYOUT"] ?? "1") != "0"
    }

    struct Table: Equatable {
        var version: Int
        var layout: String
        var expertCount: Int
        var rowsPerHalf: Int
        var rowsPerDown: Int
        var expertOrder: [Int]
        var modulePaths: [String]
        var identityOrder: Bool
    }

    struct FusedSlab {
        var weight: MLXArray
        var scales: MLXArray
        var biases: MLXArray
    }

    /// Per-module fused 3-tile slabs retained across `loadWeights` so the
    /// MoE apply path can gather gate, up, and down from one allocation.
    nonisolated(unsafe) static var fusedSlabs: [String: FusedSlab] = [:]
    nonisolated(unsafe) static var loadedTable: Table?

    static func tableExists(at directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(tableFileName).path)
    }

    static func loadTable(at directory: URL) -> Table? {
        let url = directory.appendingPathComponent(tableFileName)
        guard let data = try? Data(contentsOf: url),
            let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let version = jsonInt(root["version"]),
            version == currentVersion,
            let layout = root["layout"] as? String,
            layout == currentLayout,
            let expertCount = jsonInt(root["expert_count"]),
            let rowsPerHalf = jsonInt(root["rows_per_half"]),
            let rowsPerDown = jsonInt(root["rows_per_down"]), rowsPerDown > 0
        else { return nil }
        let order = (root["expert_order"] as? [Any])?.compactMap(jsonInt) ?? Array(0..<expertCount)
        let paths = (root["layers"] as? [[String: Any]] ?? []).compactMap {
            $0["module_path"] as? String
        }
        let identity = order.elementsEqual(0..<expertCount)
        return Table(
            version: version,
            layout: layout,
            expertCount: expertCount,
            rowsPerHalf: rowsPerHalf,
            rowsPerDown: rowsPerDown,
            expertOrder: order,
            modulePaths: paths,
            identityOrder: identity
        )
    }

    /// Swap each listed SwitchGLU to the fused `gate_up_proj` topology so
    /// `loadWeights` can bind the first two tiles; `down_proj` is a view of
    /// the third tile of the same slab.
    ///
    /// One combined `update(modules:)` call, never one per path. A single
    /// path at layer index >= 1 unflattens to a `.none`-padded `layers`
    /// array (ModuleChildren.unflattened pads missing indices), and
    /// `update(modules:)` traps on the leading `.none` — the ranked
    /// resident's boot crash (Module.swift:598,
    /// `unexpectedStructure(key: "layers")`). With every listed layer in one
    /// call, each index is a dictionary and the update recurses. The table
    /// is generated from the checkpoint's own `switch_mlp` stems, so the
    /// listed indices are contiguous from zero.
    static func fuseListedSwitchGLUs(in model: Module, table: Table) {
        let listed = Set(table.modulePaths)
        var pairs = [(String, Module)]()
        for (modulePath, module) in model.namedModules() {
            guard listed.contains(modulePath), let glu = module as? SwitchGLU,
                !glu.hasFusedGateUp
            else { continue }
            pairs.append((modulePath, glu.fusingGateUp()))
        }
        guard !pairs.isEmpty else { return }
        model.update(modules: ModuleChildren.unflattened(pairs))
    }

    static func shouldFuseLoad(at directory: URL) -> Bool {
        enabled && loadTable(at: directory) != nil
    }

    static func slab(for modulePath: String) -> FusedSlab? {
        fusedSlabs[modulePath]
    }

    /// Load a 3-tile fused checkpoint: read shards, remap `gate_up_down_proj`
    /// into `gate_up_proj` + `down_proj` views of the same slab, then quantize
    /// and bind. Old-version tables never reach here (`shouldFuseLoad`).
    static func loadFusedWeights(
        modelDirectory: URL,
        model: Qwen4ExpModel,
        table: Table,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws {
        fusedSlabs = [:]
        loadedTable = table

        var shardURLs: [URL] = []
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator where url.pathExtension == "safetensors" {
            shardURLs.append(url)
        }
        shardURLs.sort { $0.lastPathComponent < $1.lastPathComponent }

        var weights = [String: MLXArray]()
        var metadata = [String: String]()
        for (index, url) in shardURLs.enumerated() {
            var (w, m) = try loadArraysAndMetadata(url: url)
            w = w.filter { model.shouldLoadWeight(named: $0.key) }
            if !w.isEmpty {
                eval(Array(w.values))
            }
            for (key, value) in w { weights[key] = value }
            if metadata.isEmpty { metadata = m }
            TrackBootPhase.mark("shard \(index + 1)/\(shardURLs.count) resident")
        }

        weights = model.sanitize(weights: weights, metadata: metadata)
        splitFusedTensors(&weights, table: table)

        if let perLayerQuantization {
            quantize(model: model) { path, _ in
                guard weights["\(path).scales"] != nil else { return nil }
                return resolveQuantization(
                    path: path, perLayerQuantization: perLayerQuantization,
                    aliasing: model)?.asTuple
            }
        }

        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: [.all])
        TrackBootPhase.mark("strict parameter update verified")
        eval(model)
    }

    static func splitFusedTensors(_ weights: inout [String: MLXArray], table: Table) {
        let marker = ".gate_up_down_proj."
        var stems = Set<String>()
        for key in weights.keys where key.contains(marker) {
            let range = key.range(of: marker)!
            stems.insert(String(key[..<range.lowerBound]))
        }
        for stem in stems.sorted() {
            let wKey = "\(stem).gate_up_down_proj.weight"
            let sKey = "\(stem).gate_up_down_proj.scales"
            let bKey = "\(stem).gate_up_down_proj.biases"
            guard let weight = weights.removeValue(forKey: wKey),
                let scales = weights.removeValue(forKey: sKey),
                let biases = weights.removeValue(forKey: bKey)
            else { continue }
            fusedSlabs[stem] = FusedSlab(weight: weight, scales: scales, biases: biases)
            let wViews = tileViews(weight, table: table)
            let sViews = tileViews(scales, table: table)
            let bViews = tileViews(biases, table: table)
            weights["\(stem).gate_up_proj.weight"] = wViews.gateUp
            weights["\(stem).gate_up_proj.scales"] = sViews.gateUp
            weights["\(stem).gate_up_proj.biases"] = bViews.gateUp
            weights["\(stem).down_proj.weight"] = wViews.down
            weights["\(stem).down_proj.scales"] = sViews.down
            weights["\(stem).down_proj.biases"] = bViews.down
        }
    }

    static func tileViews(_ fused: MLXArray, table: Table) -> (gateUp: MLXArray, down: MLXArray) {
        let expertCount = fused.dim(0)
        let rowsPerHalf = table.rowsPerHalf
        let rowsPerDown = table.rowsPerDown
        let inner = fused.dim(2)
        let tile = rowsPerHalf * inner
        let downInner = tile / rowsPerDown
        let expertStride = 3 * tile
        let gateUp = asStrided(
            fused, [expertCount, 2 * rowsPerHalf, inner],
            strides: [expertStride, inner, 1], offset: 0)
        let down = asStrided(
            fused, [expertCount, rowsPerDown, downInner],
            strides: [expertStride, downInner, 1], offset: 2 * tile)
        return (gateUp, down)
    }

    private static func jsonInt(_ value: Any?) -> Int? {
        if let integer = value as? Int { return integer }
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            !CFNumberIsFloatType(number)
        else { return nil }
        return Int(number.stringValue)
    }
}

extension Qwen4ExpModel: QuantizationPathAliasing {
    public func quantizationPathAliases(for path: String) -> [String] {
        let fusedSuffix = ".gate_up_down_proj"
        if path.hasSuffix(fusedSuffix) {
            let base = String(path.dropLast(fusedSuffix.count))
            return qwen35GateUpQuantizationAliases(for: "\(base).gate_up_proj")
                + qwen35GateUpQuantizationAliases(for: "\(base).down_proj")
        }
        return qwen35GateUpQuantizationAliases(for: path)
    }
}
