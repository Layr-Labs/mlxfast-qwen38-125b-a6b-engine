import Foundation
import MLXFastCore

// Head delivery for the native-MTP track.
//
// OPERATOR-RATIFIED 2026-08-14. The MTP head is part of the competitive
// surface: a submission may bring its own. It declares one in
// `mtp-head.manifest.json`, which is an editable path, and the RUNNER resolves
// that declaration pre-sandbox the way it resolves a hidden golden -- refuse on
// oversize. A declared digest is parsed and carried when present but is NOT
// verified against the head bytes and is not a gate. Per DECIDE-2 (Q-B) a
// bring-your-own head is NOT required to declare one, and a digestless head and
// a wrong-digest head are treated identically: both are bounded by SIZE ONLY.
//
// THE SAFETY ARGUMENT, in one line: a head only PROPOSES tokens. The
// organizer-pinned target decides every emitted token and the trusted parent
// re-checks the whole stream against a hidden serial trajectory after the clock
// stops, so a substituted head moves the accept rate -- which is the game --
// and cannot move the output.
//
// THIS FILE IS TRUSTED CODE. It parses the declaration and applies the size
// gate; it never decides whether a run passes.
//
// ONE ARM, ONE HEAD, NO STAGING (qwen3.8-125b-a6b-mlx-v1). Two things this
// parser used to carry are gone:
//
//   * THE SECOND HEAD. The DFlash arm was excised, so `dflash-head.manifest.
//     json` and the head-KIND parameter that let one parser serve two
//     manifests are both gone. There is one manifest.
//   * THE `arm` SELECTION KEY. With one arm there is nothing to select. The
//     key, its vocabulary enum and its fail-closed parse left with the arm;
//     `arm` is now simply an unread key like any other unknown key.
//
// THE HEAD IS EMBEDDED, SO NOTHING IS STAGED OR MEASURED HERE. On this track
// the MTP head ships INSIDE the pinned target checkpoint (76 tensors under
// `language_model.mtp.*`), so there is no head directory to stage, digest, or
// measure, and no organizer head artifact to fetch. `mtp-head.manifest.json`
// survives as the DECLARATION SURFACE ONLY, and `source: "pinned"` means "the
// head embedded in the pinned target checkpoint". This file parses and refuses
// exactly as before; it just no longer describes a directory.

/// One parsed head declaration (`mtp-head.manifest.json`).
public struct Gemma4MTPHeadDeclaration: Equatable, Sendable {
    public enum Source: String, Equatable, CaseIterable, Sendable {
        /// The head embedded in the organizer's pinned target checkpoint. The
        /// default, and what an ABSENT manifest means.
        case pinned
        /// Fetched by the runner from `sourceURL` (`hf:<repo>@<rev>` or
        /// `r2:<key>`) into the run's private directory.
        case remote
        /// Shipped in the submission under the editable weights directory
        /// named by `path`.
        case inBranch = "in_branch"
    }

    public let source: Source
    public let sourceURL: String?
    public let path: String?
    public let sha256: String?
    public let bytes: Int
    public let maxBytes: Int

    public init(
        source: Source,
        sourceURL: String? = nil,
        path: String? = nil,
        sha256: String? = nil,
        bytes: Int = 0,
        maxBytes: Int = Gemma4MTPHeadDeclaration.defaultMaxBytes
    ) {
        self.source = source
        self.sourceURL = sourceURL
        self.path = path
        self.sha256 = sha256
        self.bytes = bytes
        self.maxBytes = maxBytes
    }

    /// 2 GiB. Mirrored in `mtp-head.manifest.json` (`max_bytes`) and in the
    /// contract manifest's `editableSurfaceByteBudget.exemptPathMaxBytes`; a
    /// declaration may lower it and may not raise it.
    public static let defaultMaxBytes = 2_147_483_648

    /// The default when no manifest exists at all.
    public static let pinnedDefault = Gemma4MTPHeadDeclaration(source: .pinned)

    public static let relativePath = "mtp-head.manifest.json"

    /// How a refusal names this head.
    public static let declarationNoun = "MTP"

    /// Parse a declaration, FAILING CLOSED on anything malformed.
    ///
    /// Only two things select the pinned head: the file being absent, and an
    /// explicit `source: "pinned"`. A manifest that is present but unreadable,
    /// unparseable, or internally inconsistent is a REFUSAL -- never a silent
    /// fall back -- because "your head declaration was broken so we quietly
    /// scored you on the pinned head" is exactly the failure mode that makes a
    /// leaderboard number unattributable.
    public static func parse(
        contentsOf url: URL
    ) throws -> Gemma4MTPHeadDeclaration {
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(url.path) "
                    + "exists but could not be read; refusing to fall back to "
                    + "the pinned head")
        }
        return try parse(data: data, origin: url.path)
    }

    public static func parse(
        data: Data,
        origin: String
    ) throws -> Gemma4MTPHeadDeclaration {
        let noun = declarationNoun
        guard let root = (try? JSONSerialization.jsonObject(with: data))
            as? [String: Any]
        else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) is not a JSON object")
        }
        let rawSource = (root["source"] as? String) ?? Source.pinned.rawValue
        guard let source = Source(rawValue: rawSource) else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) names an unknown source "
                    + "'\(rawSource)'; expected one of "
                    + Source.allCases.map(\.rawValue).joined(separator: ", "))
        }
        let maxBytes = try integerField(root["max_bytes"], named: "max_bytes", origin: origin)
            ?? defaultMaxBytes
        guard maxBytes > 0, maxBytes <= defaultMaxBytes else {
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) sets max_bytes "
                    + "\(maxBytes); it must be positive and may not exceed the "
                    + "track cap \(defaultMaxBytes)")
        }
        let sourceURL = root["source_url"] as? String
        let path = root["path"] as? String
        let sha256 = (root["sha256"] as? String)?.lowercased()
        let bytes = try integerField(root["bytes"], named: "bytes", origin: origin) ?? 0

        // REQUANT-ONLY (David ruling, 2026-08-26). `pinned` is the ONLY source
        // this track accepts. The head is the organizer's own weights; a
        // participant may declare a re-quantization of them and may not
        // substitute weights of their own. `remote` and `in_branch` are the two
        // spellings of "load bytes the participant chose", so both are named
        // refusals rather than gated flows.
        //
        // WHY THE CASES STAY IN THE ENUM. Deleting them would make a manifest
        // that names one fail as "unknown source", which reads like a typo. The
        // participant did not typo; they used a mode this track retired, and the
        // refusal should say so and say what replaced it.
        switch source {
        case .pinned:
            break
        case .remote:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'remote'; this track accepts source 'pinned' only. The "
                    + "\(noun) head is embedded in the organizer's pinned target "
                    + "checkpoint (fixtures/qwen3_8_125b_a6b_track.json), and "
                    + "custom head weights are not accepted. Declare a "
                    + "re-quantization of the pinned head instead")
        case .inBranch:
            throw MLXFastError.invalidInput(
                "the \(noun) head declaration at \(origin) selects source "
                    + "'in_branch'; this track accepts source 'pinned' only. A "
                    + "submission carries no head weights -- the head rides in "
                    + "the pinned target checkpoint -- so there is nothing an "
                    + "in-branch path could name. Declare a re-quantization of "
                    + "the pinned head instead")
        }

        // THE SIZE GATE SURVIVES THE NARROWING. It used to be reachable only
        // for a non-pinned source, which after the ruling would make it dead
        // code -- and deleting it would silently drop the one numeric bound a
        // declaration still carries. A `pinned` declaration may state the
        // `bytes` of the head it expects (a re-quantized head is smaller than
        // the shipped one, and stating it is how a participant records what they
        // expect), so the bound is "if you state a size, it must fit the cap"
        // rather than "non-pinned sources must state one". `bytes: 0` means
        // "not stated", which is what the checked-in declaration says.
        if bytes != 0 {
            guard bytes > 0 else {
                throw MLXFastError.invalidInput(
                    "the \(noun) head declaration at \(origin) states bytes "
                        + "\(bytes); a stated byte count must be positive")
            }
            guard bytes <= maxBytes else {
                throw MLXFastError.invalidInput(
                    "the declared \(noun) head is \(bytes) bytes, above the "
                        + "\(maxBytes)-byte cap in \(origin)")
            }
        }
        return Gemma4MTPHeadDeclaration(
            source: source,
            sourceURL: sourceURL,
            path: path,
            sha256: sha256,
            bytes: bytes,
            maxBytes: maxBytes
        )
    }

    /// A STATED NUMBER MUST BE A JSON INTEGER. `(root[key] as? NSNumber)?.intValue`
    /// drops a string or an array back onto the default -- so a declaration that
    /// says `"max_bytes": "2 GiB"` reads as the full track cap instead of
    /// refusing -- and it reads JSON `true` as 1, which walks straight into the
    /// one-byte size gate. An absent key still means "not stated" and keeps its
    /// default; a present key that is not a whole number in `Int` range refuses
    /// by name. JSON `null` is a PRESENT key whose value is not an integer, so
    /// it refuses too: only an absent key means "not stated".
    private static func integerField(
        _ raw: Any?,
        named name: String,
        origin: String
    ) throws -> Int? {
        guard let raw else {
            return nil
        }
        guard let number = raw as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            let value = number as? Int
        else {
            throw MLXFastError.invalidInput(
                "the \(declarationNoun) head declaration at \(origin) sets \(name) to "
                    + "a value that is not a JSON integer; \(name) must be a whole "
                    + "number")
        }
        return value
    }

    /// Read the declaration next to a contract root, treating ABSENCE as the
    /// pinned default and everything else as parse-or-refuse.
    public static func resolve(
        contractRoot: URL
    ) throws -> Gemma4MTPHeadDeclaration {
        let url = contractRoot.appendingPathComponent(relativePath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return pinnedDefault
        }
        return try parse(contentsOf: url)
    }
}
