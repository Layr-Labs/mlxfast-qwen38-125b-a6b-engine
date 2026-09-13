import CoreFoundation
import Foundation
import MLXFastCore

/// Offline expert-weight layout for the Qwen 3.8 125B A6B text tower.
///
/// Each routed expert's `switch_mlp.gate_proj`, `up_proj`, and `down_proj`
/// tiles (packed codes, scales, biases) are stored as one contiguous block
/// `[gate_rows | up_rows | down_as_gate_rows]` on the expert axis. Tile bytes
/// are copied, never requantized (`docs/participant-contract.md` §3.4).
/// Presence of `mlxfast-expert-layout.json` version 2 is the loader's fused
/// 3-tile signal. A missing table or an older version falls back to the
/// legacy split `gate_proj` / `up_proj` / `down_proj` layout.
enum ExpertLayout {
    static let tableFileName = "mlxfast-expert-layout.json"
    static let layoutKind = "gate_up_down_per_expert"
    static let tableVersion = 2
    static let tileCount = 3

    struct Table: Equatable {
        var version: Int
        var layout: String
        var expertCount: Int
        var rowsPerHalf: Int
        var rowsPerDown: Int
        var expertOrder: [Int]
        var layers: [Layer]

        struct Layer: Equatable {
            var checkpointPrefix: String
            var modulePath: String
            var gateRowOffset: Int
            var upRowOffset: Int
            var downRowOffset: Int
        }
    }

    struct RepackResult: Equatable {
        var keptKeys: Set<String>
        var fusedWeightMap: [String: String]
        var tensorCountDelta: Int
        var table: Table
    }

    struct Projection: Equatable {
        var stem: String
        var half: String
        var leaf: String
    }

    /// `language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight` → parts.
    static func parseProjection(_ name: String) -> Projection? {
        let marker = ".switch_mlp."
        guard let range = name.range(of: marker) else { return nil }
        let stem = String(name[..<range.upperBound].dropLast())
        let rest = String(name[range.upperBound...])
        guard let dot = rest.firstIndex(of: ".") else { return nil }
        let half = String(rest[..<dot])
        let leaf = String(rest[rest.index(after: dot)...])
        guard half == "gate_proj" || half == "up_proj" || half == "down_proj" else { return nil }
        guard leaf == "weight" || leaf == "scales" || leaf == "biases" else { return nil }
        return Projection(stem: stem, half: half, leaf: leaf)
    }

    static func fusedName(stem: String, leaf: String) -> String {
        "\(stem).gate_up_down_proj.\(leaf)"
    }

    static func modulePath(fromCheckpointStem stem: String) -> String {
        let prefix = "language_model."
        if stem.hasPrefix(prefix) {
            return String(stem.dropFirst(prefix.count))
        }
        return stem
    }

    /// Interleave per-expert tiles: `out[e] = gate[e] || up[e] || down[e]`.
    /// `down` may use a different row/inner shape; its per-expert byte count
    /// must match `gate` (this model's affine 4-bit tiles are equal-sized).
    static func interleave(
        gate: Data,
        up: Data,
        down: Data,
        expertCount: Int,
        rowsPerExpert: Int,
        inner: Int,
        byteWidth: Int
    ) throws -> Data {
        let expertBytes = try sliceByteCount(
            rows: rowsPerExpert, inner: inner, byteWidth: byteWidth, name: "expert tile")
        let expected = try multiplying(expertCount, expertBytes, name: "split expert payload")
        guard gate.count == expected, up.count == expected, down.count == expected else {
            throw MLXFastError.invalidInput(
                "expert layout split payloads are \(gate.count), \(up.count), \(down.count) bytes, expected \(expected)"
            )
        }
        var output = Data(count: try multiplying(tileCount, expected, name: "fused expert payload"))
        output.withUnsafeMutableBytes { dest in
            gate.withUnsafeBytes { gateBytes in
                up.withUnsafeBytes { upBytes in
                    down.withUnsafeBytes { downBytes in
                        for expert in 0..<expertCount {
                            let src = expert * expertBytes
                            let dst = expert * tileCount * expertBytes
                            dest.baseAddress!.advanced(by: dst).copyMemory(
                                from: gateBytes.baseAddress!.advanced(by: src),
                                byteCount: expertBytes
                            )
                            dest.baseAddress!.advanced(by: dst + expertBytes).copyMemory(
                                from: upBytes.baseAddress!.advanced(by: src),
                                byteCount: expertBytes
                            )
                            dest.baseAddress!.advanced(by: dst + 2 * expertBytes).copyMemory(
                                from: downBytes.baseAddress!.advanced(by: src),
                                byteCount: expertBytes
                            )
                        }
                    }
                }
            }
        }
        return output
    }

    /// Inverse of `interleave`. Tile bytes are identical to the inputs.
    static func split(
        fused: Data,
        expertCount: Int,
        rowsPerExpert: Int,
        inner: Int,
        byteWidth: Int
    ) throws -> (gate: Data, up: Data, down: Data) {
        let expertBytes = try sliceByteCount(
            rows: rowsPerExpert, inner: inner, byteWidth: byteWidth, name: "expert tile")
        let expected = try multiplying(
            tileCount, try multiplying(expertCount, expertBytes, name: "fused tile"),
            name: "fused expert payload")
        guard fused.count == expected else {
            throw MLXFastError.invalidInput(
                "expert layout fused payload is \(fused.count) bytes, expected \(expected)"
            )
        }
        let splitBytes = expected / tileCount
        var gate = Data(count: splitBytes)
        var up = Data(count: splitBytes)
        var down = Data(count: splitBytes)
        fused.withUnsafeBytes { src in
            gate.withUnsafeMutableBytes { gateBytes in
                up.withUnsafeMutableBytes { upBytes in
                    down.withUnsafeMutableBytes { downBytes in
                        for expert in 0..<expertCount {
                            let dst = expert * expertBytes
                            let fusedGate = expert * tileCount * expertBytes
                            gateBytes.baseAddress!.advanced(by: dst).copyMemory(
                                from: src.baseAddress!.advanced(by: fusedGate),
                                byteCount: expertBytes
                            )
                            upBytes.baseAddress!.advanced(by: dst).copyMemory(
                                from: src.baseAddress!.advanced(by: fusedGate + expertBytes),
                                byteCount: expertBytes
                            )
                            downBytes.baseAddress!.advanced(by: dst).copyMemory(
                                from: src.baseAddress!.advanced(by: fusedGate + 2 * expertBytes),
                                byteCount: expertBytes
                            )
                        }
                    }
                }
            }
        }
        return (gate, up, down)
    }

    /// Affine 4-bit dequant: `scale * q + bias` per group, low nibble first.
    static func dequantizeAffine4(
        packed: Data,
        scales: Data,
        biases: Data,
        expertCount: Int,
        rows: Int,
        packedInner: Int,
        groups: Int,
        groupSize: Int
    ) throws -> [Float] {
        let valuesPerWord = 8
        let inFeatures = packedInner * valuesPerWord
        guard inFeatures == groups * groupSize else {
            throw MLXFastError.invalidInput(
                "expert layout dequant packed inner \(packedInner) does not match \(groups) groups of \(groupSize)"
            )
        }
        let packedExpected = try multiplying(
            multiplying(expertCount, rows, name: "dequant rows"),
            packedInner * 4,
            name: "packed codes"
        )
        let scaleExpected = try multiplying(
            multiplying(expertCount, rows, name: "dequant rows"),
            groups * 2,
            name: "scales"
        )
        guard packed.count == packedExpected, scales.count == scaleExpected,
            biases.count == scaleExpected
        else {
            throw MLXFastError.invalidInput("expert layout dequant payload sizes do not match geometry")
        }
        var output = [Float](repeating: 0, count: expertCount * rows * inFeatures)
        packed.withUnsafeBytes { packedBytes in
            scales.withUnsafeBytes { scaleBytes in
                biases.withUnsafeBytes { biasBytes in
                    for expert in 0..<expertCount {
                        for row in 0..<rows {
                            let packedRow = ((expert * rows) + row) * packedInner
                            let groupRow = ((expert * rows) + row) * groups
                            let outRow = ((expert * rows) + row) * inFeatures
                            for group in 0..<groups {
                                let scale = bf16Float(
                                    scaleBytes.loadUnaligned(
                                        fromByteOffset: (groupRow + group) * 2, as: UInt16.self
                                    ).littleEndian)
                                let bias = bf16Float(
                                    biasBytes.loadUnaligned(
                                        fromByteOffset: (groupRow + group) * 2, as: UInt16.self
                                    ).littleEndian)
                                for k in 0..<groupSize {
                                    let feature = group * groupSize + k
                                    let word = packedBytes.loadUnaligned(
                                        fromByteOffset: (packedRow + feature / valuesPerWord) * 4,
                                        as: UInt32.self
                                    ).littleEndian
                                    let nibble = Int((word >> ((feature % valuesPerWord) * 4)) & 0xF)
                                    output[outRow + feature] = scale * Float(nibble) + bias
                                }
                            }
                        }
                    }
                }
            }
        }
        return output
    }

    static func table(
        stems: [String],
        expertCount: Int,
        rowsPerHalf: Int,
        rowsPerDown: Int,
        expertOrder: [Int]? = nil
    ) -> Table {
        let order = expertOrder ?? Array(0..<expertCount)
        return Table(
            version: tableVersion,
            layout: layoutKind,
            expertCount: expertCount,
            rowsPerHalf: rowsPerHalf,
            rowsPerDown: rowsPerDown,
            expertOrder: order,
            layers: stems.sorted().map { stem in
                Table.Layer(
                    checkpointPrefix: stem,
                    modulePath: modulePath(fromCheckpointStem: stem),
                    gateRowOffset: 0,
                    upRowOffset: rowsPerHalf,
                    downRowOffset: rowsPerHalf * 2
                )
            }
        )
    }

    static func encodeTable(_ table: Table) throws -> Data {
        let object: [String: Any] = [
            "version": table.version,
            "layout": table.layout,
            "expert_count": table.expertCount,
            "rows_per_half": table.rowsPerHalf,
            "rows_per_down": table.rowsPerDown,
            "expert_order": table.expertOrder,
            "layers": table.layers.map { layer -> [String: Any] in
                [
                    "checkpoint_prefix": layer.checkpointPrefix,
                    "module_path": layer.modulePath,
                    "gate_row_offset": layer.gateRowOffset,
                    "up_row_offset": layer.upRowOffset,
                    "down_row_offset": layer.downRowOffset,
                ]
            },
        ]
        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        )
        data.append(0x0A)
        return data
    }

    static func decodeTable(_ data: Data) throws -> Table {
        let object = try JSONSerialization.jsonObject(with: data)
        guard let root = object as? [String: Any],
            let version = jsonInt(root["version"]),
            let layout = root["layout"] as? String,
            let expertCount = jsonInt(root["expert_count"]),
            let rowsPerHalf = jsonInt(root["rows_per_half"]),
            let orderRaw = root["expert_order"] as? [Any],
            let layersRaw = root["layers"] as? [[String: Any]]
        else {
            throw MLXFastError.invalidInput("expert layout table is not a valid JSON object")
        }
        let rowsPerDown = jsonInt(root["rows_per_down"]) ?? 0
        let order = try orderRaw.map { value -> Int in
            guard let integer = jsonInt(value) else {
                throw MLXFastError.invalidInput("expert layout table expert_order is not integers")
            }
            return integer
        }
        let layers = try layersRaw.map { entry -> Table.Layer in
            guard let prefix = entry["checkpoint_prefix"] as? String,
                let path = entry["module_path"] as? String,
                let gate = jsonInt(entry["gate_row_offset"]),
                let up = jsonInt(entry["up_row_offset"])
            else {
                throw MLXFastError.invalidInput("expert layout table layer is missing fields")
            }
            let down = jsonInt(entry["down_row_offset"]) ?? 0
            return Table.Layer(
                checkpointPrefix: prefix, modulePath: path, gateRowOffset: gate, upRowOffset: up,
                downRowOffset: down)
        }
        return Table(
            version: version,
            layout: layout,
            expertCount: expertCount,
            rowsPerHalf: rowsPerHalf,
            rowsPerDown: rowsPerDown,
            expertOrder: order,
            layers: layers
        )
    }

    static func loadTable(at directory: URL) throws -> Table? {
        let url = directory.appendingPathComponent(tableFileName)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try decodeTable(Data(contentsOf: url))
    }

    /// Rewrite staged qwen4Exp shards: replace split gate/up/down with one fused slab.
    static func repackStagedQwen4Exp(
        stagingDirectory: URL,
        index: CheckpointIndex,
        selectedKeys: Set<String>,
        headers: [String: SafetensorsHeader]
    ) throws -> RepackResult? {
        let pairs = try collectPairs(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers,
            stagingDirectory: stagingDirectory
        )
        guard !pairs.isEmpty else { return nil }

        var skip = Set<String>()
        var fusedWeightMap: [String: String] = [:]
        var fusedByShard: [String: [FusedOutput]] = [:]
        var stems = Set<String>()

        for pair in pairs {
            skip.insert(pair.gateName)
            skip.insert(pair.upName)
            skip.insert(pair.downName)
            let fused = fusedName(stem: pair.stem, leaf: pair.leaf)
            fusedWeightMap[fused] = pair.destinationShard
            fusedByShard[pair.destinationShard, default: []].append(
                FusedOutput(name: fused, pair: pair))
            stems.insert(pair.stem)
        }
        let geometry = try layoutGeometry(from: pairs)
        let expertCount = geometry.expertCount
        let rowsPerHalf = geometry.rowsPerHalf
        let rowsPerDown = geometry.rowsPerDown

        let destinations = Set(fusedByShard.keys)
        let dropOnly = Set(skip.compactMap { index.weightMap[$0] }).subtracting(destinations)
        for shardName in destinations.sorted() + dropOnly.sorted() {
            guard let header = headers[shardName] else {
                throw MLXFastError.invalidInput("missing staged header for \(shardName)")
            }
            let shardURL = stagingDirectory.appendingPathComponent(shardName)
            var outputs: [OutputTensor] = []
            for name in header.tensors.keys.sorted() where !skip.contains(name) {
                guard let info = header.tensors[name] else { continue }
                outputs.append(
                    OutputTensor(
                        name: name, dtype: info.dtype, shape: info.shape, byteCount: info.byteCount,
                        source: .copy(info, shardURL, header)
                    ))
            }
            for fused in fusedByShard[shardName, default: []].sorted(by: { $0.name < $1.name }) {
                outputs.append(
                    OutputTensor(
                        name: fused.name,
                        dtype: fused.pair.dtype,
                        shape: [fused.pair.expertCount, fused.pair.rows * tileCount, fused.pair.inner],
                        byteCount: fused.pair.gate.byteCount + fused.pair.up.byteCount
                            + fused.pair.down.byteCount,
                        source: .fused(fused.pair)
                    ))
            }
            try writeShard(outputs, to: shardURL)
        }

        let kept = selectedKeys.subtracting(skip)
        let table = table(
            stems: Array(stems), expertCount: expertCount, rowsPerHalf: rowsPerHalf,
            rowsPerDown: rowsPerDown)
        try encodeTable(table).write(
            to: stagingDirectory.appendingPathComponent(tableFileName), options: .atomic)
        return RepackResult(
            keptKeys: kept,
            fusedWeightMap: fusedWeightMap,
            tensorCountDelta: fusedWeightMap.count - skip.count,
            table: table
        )
    }

    // MARK: - internals

    fileprivate struct Pair {
        var stem: String
        var leaf: String
        var dtype: String
        var expertCount: Int
        var rows: Int
        var inner: Int
        var downRows: Int
        var downInner: Int
        var byteWidth: Int
        var gateName: String
        var upName: String
        var downName: String
        var destinationShard: String
        var gate: SafetensorInfo
        var up: SafetensorInfo
        var down: SafetensorInfo
        var gateURL: URL
        var upURL: URL
        var downURL: URL
        var gateHeader: SafetensorsHeader
        var upHeader: SafetensorsHeader
        var downHeader: SafetensorsHeader
    }

    private struct FusedOutput {
        var name: String
        var pair: Pair
    }

    private struct OutputTensor {
        var name: String
        var dtype: String
        var shape: [Int]
        var byteCount: Int
        var source: Source
        enum Source {
            case copy(SafetensorInfo, URL, SafetensorsHeader)
            case fused(Pair)
        }
    }

    private static func collectPairs(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        stagingDirectory: URL
    ) throws -> [Pair] {
        var halves: [String: [String: String]] = [:]
        for key in selectedKeys {
            guard let parsed = parseProjection(key) else { continue }
            halves["\(parsed.stem)\0\(parsed.leaf)", default: [:]][parsed.half] = key
        }
        var pairs: [Pair] = []
        for key in halves.keys.sorted() {
            guard let map = halves[key], let gateName = map["gate_proj"], let upName = map["up_proj"],
                let downName = map["down_proj"]
            else { continue }
            let parsed = parseProjection(gateName)!
            guard let gateShard = index.weightMap[gateName],
                let upShard = index.weightMap[upName],
                let downShard = index.weightMap[downName],
                let gateHeader = headers[gateShard],
                let upHeader = headers[upShard],
                let downHeader = headers[downShard],
                let gate = gateHeader.tensors[gateName],
                let up = upHeader.tensors[upName],
                let down = downHeader.tensors[downName]
            else {
                throw MLXFastError.invalidInput("expert layout missing tensor metadata for \(gateName)")
            }
            guard gate.dtype == up.dtype, gate.shape == up.shape, gate.shape.count == 3 else {
                throw MLXFastError.invalidInput(
                    "expert layout \(gateName) / \(upName) are not matching rank-3 tiles"
                )
            }
            guard down.dtype == gate.dtype, down.shape.count == 3,
                down.shape[0] == gate.shape[0], down.byteCount == gate.byteCount
            else {
                throw MLXFastError.invalidInput(
                    "expert layout \(downName) is not a rank-3 tile with the same per-expert bytes as \(gateName)"
                )
            }
            let dtype = try TensorDType.parse(gate.dtype)
            pairs.append(
                Pair(
                    stem: parsed.stem,
                    leaf: parsed.leaf,
                    dtype: gate.dtype,
                    expertCount: gate.shape[0],
                    rows: gate.shape[1],
                    inner: gate.shape[2],
                    downRows: down.shape[1],
                    downInner: down.shape[2],
                    byteWidth: dtype.byteWidth,
                    gateName: gateName,
                    upName: upName,
                    downName: downName,
                    destinationShard: gateShard,
                    gate: gate,
                    up: up,
                    down: down,
                    gateURL: stagingDirectory.appendingPathComponent(gateShard),
                    upURL: stagingDirectory.appendingPathComponent(upShard),
                    downURL: stagingDirectory.appendingPathComponent(downShard),
                    gateHeader: gateHeader,
                    upHeader: upHeader,
                    downHeader: downHeader
                ))
        }
        return pairs
    }

    /// Weight-tile geometry per stem. Scales and biases of that stem must
    /// share `expertCount`, gate rows, and down rows. Every stem must agree
    /// with the others: the offset table stores one `rows_per_down`.
    private static func layoutGeometry(from pairs: [Pair]) throws -> (
        expertCount: Int, rowsPerHalf: Int, rowsPerDown: Int
    ) {
        var byStem: [String: [Pair]] = [:]
        for pair in pairs {
            byStem[pair.stem, default: []].append(pair)
        }
        var expertCount: Int?
        var rowsPerHalf: Int?
        var rowsPerDown: Int?
        for stem in byStem.keys.sorted() {
            let group = byStem[stem]!
            let canonical = group.first(where: { $0.leaf == "weight" }) ?? group[0]
            for pair in group {
                guard pair.expertCount == canonical.expertCount,
                    pair.rows == canonical.rows,
                    pair.downRows == canonical.downRows
                else {
                    throw MLXFastError.invalidInput(
                        "expert layout \(stem) \(pair.leaf) rows do not match the weight tile"
                    )
                }
            }
            if let expertCount, let rowsPerHalf, let rowsPerDown {
                guard expertCount == canonical.expertCount,
                    rowsPerHalf == canonical.rows,
                    rowsPerDown == canonical.downRows
                else {
                    throw MLXFastError.invalidInput(
                        "expert layout \(stem) geometry does not match the other layers"
                    )
                }
            } else {
                expertCount = canonical.expertCount
                rowsPerHalf = canonical.rows
                rowsPerDown = canonical.downRows
            }
        }
        guard let expertCount, let rowsPerHalf, let rowsPerDown else {
            throw MLXFastError.invalidInput("expert layout has no fused tiles")
        }
        return (expertCount, rowsPerHalf, rowsPerDown)
    }

    private static func writeShard(_ tensors: [OutputTensor], to url: URL) throws {
        let sorted = tensors.sorted { $0.name < $1.name }
        var headerObject: [String: Any] = [:]
        var cursor = 0
        for tensor in sorted {
            let end = try adding(cursor, tensor.byteCount, name: tensor.name)
            headerObject[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }
        var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
        while !header.count.isMultiple(of: 8) {
            header.append(0x20)
        }
        let temporary = url.deletingLastPathComponent().appendingPathComponent(
            ".\(url.lastPathComponent).expert-layout-\(UUID().uuidString)"
        )
        try Data().write(to: temporary, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: temporary)
        var published = false
        defer {
            try? output.close()
            if !published {
                try? FileManager.default.removeItem(at: temporary)
            }
        }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)
        for tensor in sorted {
            switch tensor.source {
            case .copy(let info, let source, let header):
                try copyTensor(info, from: source, header: header, to: output)
            case .fused(let pair):
                try writeFused(pair, to: output)
            }
        }
        try output.synchronize()
        try output.close()
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        try FileManager.default.moveItem(at: temporary, to: url)
        published = true
    }

    private static func copyTensor(
        _ info: SafetensorInfo,
        from source: URL,
        header: SafetensorsHeader,
        to output: FileHandle
    ) throws {
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let (offset, overflow) = header.dataBaseOffset.addingReportingOverflow(UInt64(info.dataStart))
        guard !overflow else {
            throw MLXFastError.invalidInput("expert layout copy offset overflows for \(info.name)")
        }
        try input.seek(toOffset: offset)
        try copyFileBytes(from: input, to: output, count: info.byteCount)
    }

    fileprivate static func writeFused(_ pair: Pair, to output: FileHandle) throws {
        let expertBytes = try sliceByteCount(
            rows: pair.rows, inner: pair.inner, byteWidth: pair.byteWidth, name: pair.gateName)
        let downBytes = try sliceByteCount(
            rows: pair.downRows, inner: pair.downInner, byteWidth: pair.byteWidth,
            name: pair.downName)
        guard downBytes == expertBytes else {
            throw MLXFastError.invalidInput(
                "expert layout \(pair.downName) tile is \(downBytes) bytes, expected \(expertBytes)"
            )
        }
        let gate = try FileHandle(forReadingFrom: pair.gateURL)
        let up = try FileHandle(forReadingFrom: pair.upURL)
        let down = try FileHandle(forReadingFrom: pair.downURL)
        defer {
            try? gate.close()
            try? up.close()
            try? down.close()
        }
        let (gateOff, gateOverflow) = pair.gateHeader.dataBaseOffset.addingReportingOverflow(
            UInt64(pair.gate.dataStart))
        let (upOff, upOverflow) = pair.upHeader.dataBaseOffset.addingReportingOverflow(
            UInt64(pair.up.dataStart))
        let (downOff, downOverflow) = pair.downHeader.dataBaseOffset.addingReportingOverflow(
            UInt64(pair.down.dataStart))
        guard !gateOverflow, !upOverflow, !downOverflow else {
            throw MLXFastError.invalidInput("expert layout fused offset overflows for \(pair.gateName)")
        }
        try gate.seek(toOffset: gateOff)
        try up.seek(toOffset: upOff)
        try down.seek(toOffset: downOff)
        for _ in 0..<pair.expertCount {
            try copyFileBytes(from: gate, to: output, count: expertBytes)
            try copyFileBytes(from: up, to: output, count: expertBytes)
            try copyFileBytes(from: down, to: output, count: expertBytes)
        }
    }

    private static func copyFileBytes(from input: FileHandle, to output: FileHandle, count: Int) throws {
        var remaining = count
        let chunk = 8 * 1024 * 1024
        while remaining > 0 {
            let copied = try autoreleasepool { () throws -> Int in
                let data = input.readData(ofLength: min(chunk, remaining))
                if data.isEmpty {
                    throw MLXFastError.invalidInput("unexpected EOF while rewriting expert layout")
                }
                try output.write(contentsOf: data)
                return data.count
            }
            remaining -= copied
        }
    }

    private static func sliceByteCount(rows: Int, inner: Int, byteWidth: Int, name: String) throws -> Int {
        try multiplying(multiplying(rows, inner, name: name), byteWidth, name: name)
    }

    private static func multiplying(_ a: Int, _ b: Int, name: String) throws -> Int {
        let result = a.multipliedReportingOverflow(by: b)
        guard !result.overflow, result.partialValue >= 0 else {
            throw MLXFastError.invalidInput("expert layout size overflows for \(name)")
        }
        return result.partialValue
    }

    private static func adding(_ a: Int, _ b: Int, name: String) throws -> Int {
        let result = a.addingReportingOverflow(b)
        guard !result.overflow else {
            throw MLXFastError.invalidInput("expert layout offset overflows for \(name)")
        }
        return result.partialValue
    }

    private static func jsonInt(_ value: Any?) -> Int? {
        if let integer = value as? Int { return integer }
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            !CFNumberIsFloatType(number)
        else { return nil }
        return Int(number.stringValue)
    }

    private static func bf16Float(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }
}
