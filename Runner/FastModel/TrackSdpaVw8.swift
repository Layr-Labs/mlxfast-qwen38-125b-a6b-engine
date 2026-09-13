// TrackSdpaVw8.swift
//
// Width-8 V-loop loads in the vendored SDPA vector kernel
// (`sdpa_vector_2pass_1`). TRACK_SDPA_VW8 default ON; `0` restores the
// vec4/scalar V loads. The K-loop and the add order do not change.
//
// The kernel takes the wide load only when each thread owns 8 V elements
// (value dim 256) and both V strides are 32-byte multiples, so every thread
// in the walk sees a 32-byte-contiguous row.

import Foundation

enum TrackSdpaVw8 {
    static let simdGroup = 32
    static let width = 8
    static let alignmentBytes = 32

    static func enabledValue(_ raw: String?) -> Bool {
        (raw ?? "1") != "0"
    }

    static func isEnabled(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        enabledValue(environment["TRACK_SDPA_VW8"])
    }

    /// Shape-safe for a width-8 V vector load. Matches the in-kernel gate.
    static func allowsWidth8(
        valueDim: Int,
        valueHeadStride: Int,
        valueSeqStride: Int,
        dtypeBytes: Int
    ) -> Bool {
        guard valueDim == simdGroup * width else { return false }
        guard dtypeBytes > 0 else { return false }
        guard valueHeadStride > 0, valueSeqStride > 0 else { return false }
        return (valueHeadStride * dtypeBytes).isMultiple(of: alignmentBytes)
            && (valueSeqStride * dtypeBytes).isMultiple(of: alignmentBytes)
    }

    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    static var kernelHeaderURL: URL {
        repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h")
    }

    static var dispatchSourceURL: URL {
        repositoryRoot.appendingPathComponent(
            "Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/scaled_dot_product_attention.cpp")
    }
}
