// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "mlxfast-challenge-dev",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "mlxfast-swift", targets: ["MLXFastCLI"]),
        .library(name: "MLXFastCore", targets: ["MLXFastCore"]),
        .library(name: "MLXFastTransform", targets: ["MLXFastTransform"]),
        .library(name: "MLXFastHarness", targets: ["MLXFastHarness"]),
        // THE SCORED ENGINE, built HERE and not out of the submodule. Same
        // shim, same scratch path, same staged destination -- the only
        // difference is that this one registers `Runner/` over the fork's
        // built-in runner.
        //
        // NOT named `bench-worker`, and the name is the finding. The fork
        // declares an executable product of that name too, and SwiftPM 6.3.3
        // does NOT prefer the root package's: with both declared, `swift build
        // --product bench-worker` silently builds the FORK's product, and so
        // does a whole-package `swift build` -- verified twice, once with each
        // command, by linking and then reading the binary for the track
        // runner's symbols (absent both times). A same-named root product is
        // therefore not shadowed loudly, it is shadowed silently, which is the
        // one outcome a scored engine cannot have. The product carries a name
        // nothing else in the graph claims, and setup.sh names that product and
        // exports MLXFAST_BENCH_WORKER_EXECUTABLE so
        // tools/stage-bench-worker.sh -- unchanged, and still copying to the
        // FIXED .build/release/bench-worker that benchd resolves -- picks it up.
        .executable(name: "track-bench-worker", targets: ["BenchWorker"]),
        // The track's editable Runner. A PRODUCT so a consumer of this
        // package (Darkbloom, a future harness) can link the same runner the
        // scored engine serves, instead of the fork's built-in one.
        .library(name: "TrackRunner", targets: ["TrackRunner"]),
    ],
    dependencies: [
        // THE ENGINE IS THE FORK. Vendor/mlx-swift-lm is a git SUBMODULE
        // pinned to Layr-Labs/mlx-swift-lm 449f2d0 (branch
        // feat/qwen38-flash-next-runner): the Qwen 3.8 Flash-Next model port, the
        // MLXRunners scaffold and the generic `bench-worker` Engine
        // Protocol v1 server. benchd spawns that binary; this package
        // builds it as a dependency product. A DEV PIN to an unmerged
        // branch -- re-pin to fork main once the scaffold and the runner
        // merge. See docs/qwen38-125b-a6b-port-notes.md section 13.
        //
        // mlx-swift is UNCHANGED at Layr-Labs/mlx-swift 6b0505cc (MLX
        // 0.32.2), with its nested submodules copied in as plain files:
        // Source/Cmlx/mlx at 734241bb and Source/Cmlx/mlx-c at 9ff12fab.
        // It stays a VENDORED TREE, because its Metal kernel sources are
        // the track's optimization surface, and an editable path cannot be
        // a submodule. 6b0505cc is the MLX core that Darkbloom
        // (Layr-Labs/d-inference) builds the same fork code against. The
        // fork's Package.swift path-depends on a sibling ../mlx-swift when
        // one exists, so one runner must have ONE core: a different
        // vendored commit here would compile the fork against a core its
        // own repository never tested.
        // The fork declares its own mlx-swift dependency as a floating
        // `branch: "main"` URL; SwiftPM resolves a ROOT path dependency of
        // the same package identity ahead of it, so every target in the
        // graph -- the fork's included -- builds against Vendor/mlx-swift.
        // SwiftPM reports that override as a "conflicting identity"
        // warning. See docs/qwen38-125b-a6b-port-notes.md section 13.
        .package(path: "Vendor/mlx-swift"),
        .package(path: "Vendor/mlx-swift-lm"),
        // The resolved dependency graph is frozen. In the engine repository
        // the enforcement that survives is the one that lives here: setup.sh
        // refuses to build over a Package.swift/Package.resolved that differs
        // from the committed state, and every build and resolve passes
        // --force-resolved-versions so SwiftPM fails closed instead of
        // silently re-resolving. The byte-verification of this manifest
        // against a trusted reference runs from .github/scripts/, which THIS
        // tree carries: submission-static-review-checks.sh is the gate, and
        // the roster it verifies includes this file.
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.3.3"),
    ],
    targets: [
        .target(name: "MLXFastCore"),
        .target(
            name: "MLXFastTransform",
            dependencies: ["MLXFastCore"]
        ),
        // The trusted-harness source scope is this manifest,
        // Package.resolved, Sources/MLXFastCLI, Sources/MLXFastTrustedHarness
        // and Sources/MLXFastCore. It is declared here so a submission cannot
        // expand or repoint the targets feeding the trusted binary without
        // that showing up as a manifest diff. The byte-verification of this
        // scope against trusted git content ran from .github/scripts/, which
        // this engine repository no longer carries; enforcing it is the
        // grading pipeline's job.
        .target(
            name: "MLXFastHarness",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXFastTrustedHarness",
            swiftSettings: [
                .define("MLXFAST_TRUSTED_HARNESS")
            ]
        ),
        .executableTarget(
            name: "MLXFastCLI",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // THE RUNNER IS EDITABLE, AND IT LIVES HERE. `Runner/` is a copy of
        // the fork's `Libraries/MLXRunners/Qwen4ExpRunner.swift` at the
        // pinned commit 449f2d01, renamed to `TrackQwen4ExpRunner` so both can
        // be linked at once, with its manifest unchanged byte for byte -- the
        // runner manifest digest is a benchd conformance input and must not
        // move when the code behind it does.
        //
        // The fork stays a PINNED SUBMODULE for the engine core. Shadowing is
        // a registry fact, not a build fact: `RunnerRegistry.register` lets a
        // later registration replace an earlier claim on the same
        // `model_type`, and `Sources/BenchWorker` performs that registration
        // before anything resolves a runner.
        .target(
            name: "TrackRunner",
            dependencies: [
                .product(name: "MLXRunners", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Runner"
        ),
        // This repository's `bench-worker`: the fork's shim at 449f2d01 with
        // ONE added statement, the `TrackQwen4ExpRunner` registration. Every
        // other line, and the order they run in, is the fork's.
        .executableTarget(
            name: "BenchWorker",
            dependencies: [
                "TrackRunner",
                .product(name: "MLXRunners", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/BenchWorker",
            plugins: ["TrackBenchRevisionStamp"]
        ),
        // The build-revision stamp the shim reports on the hello. A MIRROR of
        // the fork's `Plugins/BenchRevisionStamp`, which cannot be reused from
        // here: the fork exports no plugin product and the fork is pinned.
        // This one stamps THIS repository's revision, which is the tree that
        // now carries the Runner.
        // Named `TrackBenchRevisionStamp`, not `BenchRevisionStamp`: SwiftPM
        // requires target names to be unique across the whole package graph,
        // and the fork already has one.
        .plugin(
            name: "TrackBenchRevisionStamp",
            capability: .buildTool(),
            path: "Plugins/TrackBenchRevisionStamp"
        ),
        .testTarget(
            name: "MLXFastTests",
            dependencies: [
                "MLXFastCore",
                "MLXFastTransform",
                "MLXFastHarness",
                // The head adoption seam lives in the editable Runner, so the
                // test that proves its default is bit-exact links the Runner
                // and the model family it rebuilds.
                "TrackRunner",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ]
        ),
    ]
)
