// Power-of-two wrap: `x % c` -> `x & (c - 1)` for non-negative x.
//
// GPU integer division is a software routine on Metal. Prefill windows of
// 1024 and head dim 256 are compile-time template ints; the AND form is the
// same value as unsigned modulo. `TRACK_BITMASK_RINGS=0` restores `%`.
// Non-power-of-two capacities (MTP T=3,5,6,7; GDN Hv=48) stay on modulo.

import Foundation

enum TrackBitmaskRings {
    /// Default ON. `TRACK_BITMASK_RINGS=0` restores integer modulo.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_BITMASK_RINGS"] ?? "1") != "0"
    }()

    static func isPowerOfTwo(_ capacity: Int) -> Bool {
        capacity > 0 && (capacity & (capacity - 1)) == 0
    }

    /// `x % capacity` for non-negative `x`. Bitmask iff enabled and `capacity`
    /// is a power of two; otherwise modulo. Negative `x` is refused: AND is
    /// not equal to remainder for those values.
    static func wrap(_ x: Int, _ capacity: Int) -> Int {
        wrap(x, capacity, on: enabled)
    }

    static func wrap(_ x: Int, _ capacity: Int, on: Bool) -> Int {
        precondition(x >= 0, "TrackBitmaskRings.wrap: index must be non-negative")
        precondition(capacity > 0, "TrackBitmaskRings.wrap: capacity must be positive")
        if on && isPowerOfTwo(capacity) {
            return x & (capacity - 1)
        }
        return x % capacity
    }

    /// Metal helper body. Toggle off is a plain `x % c`.
    static func metalHelperSource(_ on: Bool) -> String {
        if on {
            return """
                METAL_FUNC uint track_ring(uint x, uint c) {
                    return ((c & (c - 1u)) == 0u) ? (x & (c - 1u)) : (x % c);
                }

                """
        }
        return """
            METAL_FUNC uint track_ring(uint x, uint c) {
                return x % c;
            }

            """
    }

    /// Baked at first use from `enabled` (process env). Kernel headers read this.
    static let metalHelper: String = metalHelperSource(enabled)
}
