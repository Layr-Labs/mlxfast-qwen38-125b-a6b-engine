// TrackMemoryBudget.swift -- the runner's memory budget, pinned before the
// first Metal buffer.
//
// THE CEILING ON THIS MACHINE CLASS IS NOT RAM. macOS caps the memory the
// GPU may wire (`iogpu.wired_limit_mb`, 100 GiB of 128 GiB on the M5 Max);
// once Metal wires past it the driver stalls the whole machine and the
// kernel's watchdog resets the box, with RAM still free. The weights alone
// hold 82 GB of that budget. MLX's defaults do not know this: its buffer
// cache may grow to 0.95 x physical (a long prefill left 19 GB of released
// intermediates parked there) and nothing keeps the weights resident, so
// under pressure the kernel evicted weight pages and the GPU re-faulted them
// from SSD (15-21 s attention stalls before each reset).
//
// What this pins, the way mlx-serve and the d-inference provider do:
//   * MLX cache limit 4 GiB and MLX memory limit = wired cap - 6 GiB;
//   * a wired-memory ticket sized just under the cap, so the weights and the
//     KV stay resident;
//   * a memory-pressure source that releases the MLX cache on warning or
//     critical;
//   * the fast path's guard limit: past wired cap - 3 GiB of active + cache
//     the WORKER aborts (exit 3), never the box.
// Every knob is env-overridable in bytes; 0 keeps MLX's default for it.

import Foundation
import MLX

public enum TrackMemoryBudget {
    nonisolated(unsafe) private static var configured = false
    nonisolated(unsafe) private static var pressureSource: DispatchSourceMemoryPressure?
    nonisolated(unsafe) private static var wiredTicket: MLX.WiredMemoryTicket?

    /// `iogpu.wired_limit_mb`, the GPU-wired ceiling macOS enforces.
    public static func gpuWiredLimitBytes() -> Int? {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("iogpu.wired_limit_mb", &value, &size, nil, 0) == 0, value > 0 else {
            return nil
        }
        return Int(value) << 20
    }

    /// Pin the budget once per process. Call before the model loads.
    public static func configureOnce() async {
        if configured { return }
        configured = true
        let environment = ProcessInfo.processInfo.environment
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let gpuWired = gpuWiredLimitBytes() ?? (physical * 3 / 4)
        let cacheLimit = Int(environment["MLXFAST_MLX_CACHE_LIMIT_BYTES"] ?? "") ?? (4 << 30)
        let memoryLimit = Int(environment["MLXFAST_MLX_MEMORY_LIMIT_BYTES"] ?? "") ?? (gpuWired - (6 << 30))
        let wiredLimit = Int(environment["MLXFAST_MLX_WIRED_LIMIT_BYTES"] ?? "") ?? (gpuWired - (8 << 30))
        if cacheLimit > 0 { Memory.cacheLimit = cacheLimit }
        if memoryLimit > 0 { Memory.memoryLimit = memoryLimit }
        var appliedWired = 0
        if wiredLimit > 0 {
            let ticket = MLX.WiredMemoryTicket(
                size: wiredLimit, policy: MLX.WiredSumPolicy(), manager: .shared, kind: .active)
            appliedWired = await ticket.start()
            wiredTicket = ticket
        }
        TrackQwen4ExpFastModel.wiredGuardLimit = gpuWired - (3 << 30)

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
            Data(("track-runner: memory budget cache_limit=\(cacheLimit >> 20)MiB "
                + "memory_limit=\(memoryLimit >> 20)MiB wired_limit=\(appliedWired >> 20)MiB "
                + "gpu_wired_limit=\(gpuWired >> 20)MiB physical=\(physical >> 20)MiB\n").utf8))
    }
}
