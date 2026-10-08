import Foundation
import HuggingFace

public enum ModelUtils {
    public static func resolveModelType(
        repoID: Repo.ID,
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> String? {
        let modelNameComponents = repoID.name.split(separator: "/").last?.split(separator: "-")
        let modelURL = try await resolveOrDownloadModel(
            repoID: repoID,
            requiredExtension: "safetensors",
            hfToken: hfToken,
            cache: cache
        )
        let configJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: modelURL.appendingPathComponent("config.json")))
        if let config = configJSON as? [String: Any] {
            return (config["model_type"] as? String)
                ?? (config["architecture"] as? String)
                ?? (config["model_version"] as? String)
                ?? modelNameComponents?.first?.lowercased()
        }
        return nil
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - string: The repository name
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    ///   - hfToken: The huggingface token for access to gated repositories, if needed.
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        hfToken: String? = nil,
        cache: HubCache = .default
    ) async throws -> URL {
        let client: HubClient
        if let token = hfToken, !token.isEmpty {
            print("Using HuggingFace token from configuration")
            client = HubClient(host: HubClient.defaultHost, bearerToken: token, cache: cache)
        } else {
            client = HubClient(cache: cache)
        }
        let resolvedCache = client.cache ?? cache
        return try await resolveOrDownloadModel(
            client: client,
            cache: resolvedCache,
            repoID: repoID,
            requiredExtension: requiredExtension,
            additionalMatchingPatterns: additionalMatchingPatterns
        )
    }

    /// Resolves a model from cache or downloads it if not cached.
    /// - Parameters:
    ///   - client: The HuggingFace Hub client
    ///   - cache: The HuggingFace cache
    ///   - repoID: The repository ID
    ///   - requiredExtension: File extension that must exist for cache to be considered complete (e.g., "safetensors")
    /// - Returns: The model directory URL
    public static func resolveOrDownloadModel(
        client: HubClient,
        cache: HubCache = .default,
        repoID: Repo.ID,
        requiredExtension: String,
        additionalMatchingPatterns: [String] = [],
        progressHandler: (@MainActor @Sendable (Progress) -> Void)? = nil
    ) async throws -> URL {
        let normalizedRequiredExtension = requiredExtension.hasPrefix(".")
            ? String(requiredExtension.dropFirst())
            : requiredExtension

        // Store downloaded model snapshots under the configured Hugging Face cache root.
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        let modelDir = cache.cacheDirectory
            .appendingPathComponent("mlx-audio")
            .appendingPathComponent(modelSubdir)

        let defaults = Self.defaultDownloadPatterns(requiredExtension: normalizedRequiredExtension)
        let requested = Set(additionalMatchingPatterns)
        var recorded = recordedPatterns(modelDir: modelDir) ?? []

        // A snapshot with a non-empty weights file and a config that parses is
        // usable. Whether it is complete for THIS caller depends on the
        // patterns it was fetched with: resolveModelType pre-downloads with
        // no additional patterns, so a later load that needs e.g. "*.mvn"
        // would otherwise silently get a partial snapshot. Default patterns
        // ("*.json", "*.safetensors", ...) come with every fetch and count as
        // covered without a manifest entry.
        var usableSnapshot = false
        if FileManager.default.fileExists(atPath: modelDir.path) {
            let files = try? FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: [.fileSizeKey])
            let hasRequiredFile = files?.contains { file in
                guard file.pathExtension == normalizedRequiredExtension else { return false }
                let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
                return size > 0
            } ?? false

            if hasRequiredFile {
                // Validate that config.json is valid JSON
                let configPath = modelDir.appendingPathComponent("config.json")
                if FileManager.default.fileExists(atPath: configPath.path) {
                    if let configData = try? Data(contentsOf: configPath),
                       let _ = try? JSONSerialization.jsonObject(with: configData) {
                        if requested.isSubset(of: recorded.union(defaults)) {
                            print("Using cached model at: \(modelDir.path)")
                            return modelDir
                        }
                        usableSnapshot = true
                        // Fall through and fetch what the uncovered patterns name.
                    } else {
                        print("Cached config.json is invalid, clearing cache...")
                        Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                        recorded = []
                    }
                }
            } else {
                print("Cached model appears incomplete, clearing cache...")
                Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
                recorded = []
            }
        }

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // Ask the Hub what the patterns name. The listing is what lets a
        // pattern be recorded as complete: a pattern the repo has no file for
        // is complete the moment the Hub says so (Whisper asks for "*.model";
        // no Whisper repo has one), and the Hub client serves the cached
        // snapshot when it cannot list, so without a listing nothing new may
        // be recorded, or an offline load would mark files as present that
        // were never fetched. One request, taken at most once per snapshot
        // and set of patterns while online; afterwards the hit above answers.
        let listing = try? await client.listFiles(in: repoID, kind: .model, revision: "main", recursive: true)
        let progress: @MainActor @Sendable (Progress) -> Void = progressHandler ?? { progress in
            print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
        }

        let fetched: Set<String>
        if usableSnapshot {
            // The weights are here; fetch only what the uncovered patterns
            // name, into the same directory, so nothing already present is
            // downloaded or copied again. With no listing (offline) the
            // snapshot is served as it is, uncertified: the loader finds out
            // whether the files it wants are there, as it always did.
            let missing = requested.subtracting(recorded.union(defaults))
            guard let listing else {
                print("Using cached model at: \(modelDir.path) (the Hub could not be asked for \(missing.sorted().joined(separator: ", ")))")
                return modelDir
            }
            let absent = listing.filter { entry in
                entry.type == .file
                    && missing.contains { fnmatch($0, entry.path, 0) == 0 }
                    && !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent(entry.path).path)
            }
            if !absent.isEmpty {
                print("Fetching \(absent.count) file(s) for \(repoID): \(missing.sorted().joined(separator: ", "))...")
                _ = try await client.downloadSnapshot(
                    of: repoID,
                    kind: .model,
                    to: modelDir,
                    revision: "main",
                    matching: Array(missing),
                    progressHandler: progress
                )
            }
            fetched = missing
        } else {
            let patterns = defaults.union(requested).union(recorded)
            print("Downloading model \(repoID)...")
            _ = try await client.downloadSnapshot(
                of: repoID,
                kind: .model,
                to: modelDir,
                revision: "main",
                matching: Array(patterns),
                progressHandler: progress
            )
            fetched = requested.union(recorded)
        }

        // Post-download validation: ensure required files are non-zero
        let downloadedFiles = try? FileManager.default.contentsOfDirectory(
            at: modelDir, includingPropertiesForKeys: [.fileSizeKey]
        )
        let hasValidFile = downloadedFiles?.contains { file in
            guard file.pathExtension == normalizedRequiredExtension else { return false }
            let size = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
            return size > 0
        } ?? false

        if !hasValidFile {
            Self.clearCaches(modelDir: modelDir, repoID: repoID, hubCache: cache)
            throw ModelUtilsError.incompleteDownload(repoID.description)
        }

        print("Model downloaded to: \(modelDir.path)")
        // Only a listing proves the patterns are complete; a download that the
        // client served from its own cache proves nothing about them.
        if listing != nil {
            recordPatterns(modelDir: modelDir, patterns: recorded.union(fetched))
        }
        return modelDir
    }

    /// Patterns every snapshot download includes regardless of caller-supplied
    /// `additionalMatchingPatterns`. A cache hit counts these as covered;
    /// recording them in the manifest would be redundant.
    private static func defaultDownloadPatterns(requiredExtension: String) -> Set<String> {
        [
            "*.\(requiredExtension)",
            "*.safetensors",
            "*.json",
            "*.txt",
            "*.wav",
        ]
    }

    /// Name of the manifest recording which `additionalMatchingPatterns` a
    /// cached snapshot was downloaded with (one pattern per line).
    private static let patternsManifestName = ".mlx-audio-patterns"

    /// Patterns a cached snapshot was downloaded with, or nil when the
    /// manifest is missing/unreadable (cache predates this mechanism).
    private static func recordedPatterns(modelDir: URL) -> Set<String>? {
        let manifestURL = modelDir.appendingPathComponent(patternsManifestName)
        guard let data = try? Data(contentsOf: manifestURL),
              let text = String(data: data, encoding: .utf8)
        else { return nil }
        return Set(text.split(separator: "\n").map(String.init))
    }

    private static func recordPatterns(modelDir: URL, patterns: Set<String>) {
        let manifestURL = modelDir.appendingPathComponent(patternsManifestName)
        try? patterns.sorted().joined(separator: "\n").write(
            to: manifestURL, atomically: true, encoding: .utf8
        )
    }

    private static func clearCaches(modelDir: URL, repoID: Repo.ID, hubCache: HubCache) {
        try? FileManager.default.removeItem(at: modelDir)
        let hubRepoDir = hubCache.repoDirectory(repo: repoID, kind: .model)
        if FileManager.default.fileExists(atPath: hubRepoDir.path) {
            print("Clearing Hub cache at: \(hubRepoDir.path)")
            try? FileManager.default.removeItem(at: hubRepoDir)
        }
    }
}

public enum ModelUtilsError: LocalizedError {
    case incompleteDownload(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        }
    }
}
