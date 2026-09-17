// TrackMemoryBudget.swift -- the runner's MLX memory ceiling, pinned before
// the first Metal buffer, the way d-inference's MLXMemoryGuard and mlx-serve
// do it.
//
// The ceiling on this machine class is the memory the GPU may wire
// (`iogpu.wired_limit_mb`), not RAM. MLX's default buffer cache may grow to
// 0.95 x physical; a 32K prefill parked 19 GB of released intermediates there
// on top of the 82 GB of weights and the box reset. So:
//   * MLX memory limit = physical - 6 GiB reserve for the OS;
//   * MLX cache limit = min(0.75 x memory limit, 8 GiB);
//   * a memory-pressure source that releases the MLX cache on warning or
//     critical.
// The fast path also releases the cache after every window of at least
// `TrackQwen4ExpFastModel.cacheReleaseMinWindow` tokens (mlx-serve clears
// the cache per prefill chunk); single-token decode steps never touch it.
// Nothing is wired: a wired ticket sized near the cap starved the page cache
// and made every prefill after an idle wait re-read the n-gram rows and
// weight pages from SSD (measured 2026-09-16: 854 vs 2399 tok/s).
// Every limit is env-overridable in bytes; 0 keeps MLX's default for it.

import Foundation
import MLX

public enum TrackMemoryBudget {
    nonisolated(unsafe) private static var configured = false
    nonisolated(unsafe) private static var pressureSource: DispatchSourceMemoryPressure?

    /// Pin the budget once per process. Call before the model loads.
    public static func configureOnce() {
        if configured { return }
        configured = true
        let environment = ProcessInfo.processInfo.environment
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let memoryLimit = Int(environment["MLXFAST_MLX_MEMORY_LIMIT_BYTES"] ?? "")
            ?? max(1 << 30, physical - (6 << 30))
        let cacheLimit = Int(environment["MLXFAST_MLX_CACHE_LIMIT_BYTES"] ?? "")
            ?? min(memoryLimit * 3 / 4, 8 << 30)
        if memoryLimit > 0 { Memory.memoryLimit = memoryLimit }
        if cacheLimit > 0 { Memory.cacheLimit = cacheLimit }

        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.warning, .critical],
            queue: DispatchQueue(label: "track-runner.memory-pressure"))
        source.setEventHandler {
            let level = source.data.contains(.critical) ? "critical" : "warning"
            let before = Memory.cacheMemory
            Memory.clearCache()
            FileHandle.standardError.write(
                Data(("track-runner: memory pressure \(level): released \(before >> 20)MiB of MLX cache "
                    + "(active \(Memory.activeMemory >> 20)MiB)\n").utf8))
        }
        source.resume()
        pressureSource = source

        FileHandle.standardError.write(
            Data(("track-runner: memory budget memory_limit=\(memoryLimit >> 20)MiB "
                + "cache_limit=\(cacheLimit >> 20)MiB physical=\(physical >> 20)MiB\n").utf8))
    }
}
