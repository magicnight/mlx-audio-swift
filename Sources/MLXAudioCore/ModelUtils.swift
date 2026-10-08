import Foundation
import HuggingFace
import os

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
            client = HubClient(host: Self.hubHost, bearerToken: token, cache: cache)
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
        // covered without a manifest entry, and so does a pattern that names
        // one file (no glob characters) that is on disk.
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
                        if uncoveredPatterns(requested, recorded: recorded, defaults: defaults, in: modelDir).isEmpty {
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

        // A manifest beside a snapshot that could not be used (weights with
        // no config) proves nothing: dropped before the fresh fetch, so an
        // offline fetch cannot leave it behind for the next resolve to trust.
        if !usableSnapshot, !recorded.isEmpty {
            recorded = []
            forgetRecordedPatterns(modelDir: modelDir)
        }

        // Create directory if needed
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // The revision the cached snapshot came from, so the listing and the
        // files fetched to complete it match the weights already on disk; a
        // fresh snapshot follows `main`.
        let revision = usableSnapshot
            ? (cache.resolveRevision(repo: repoID, kind: .model, ref: "main") ?? "main")
            : "main"

        // Ask the Hub what the patterns name. The listing is what lets a
        // pattern be recorded as complete: a pattern the repo has no file for
        // is complete the moment the Hub says so (Whisper asks for "*.model";
        // no Whisper repo has one), and the Hub client serves the cached
        // snapshot when it cannot list, so without a listing nothing new may
        // be recorded, or an offline load would mark files as present that
        // were never fetched. One request, bounded in time (two when the Hub
        // no longer has the cached commit and `main` is tried), not repeated
        // for a while after it ran out of time or took seconds to fail, and
        // taken at most once per snapshot and set of patterns while the Hub
        // answers; afterwards the hit above answers. Only when there is
        // something to certify: a fresh fetch of the default patterns alone
        // has no pattern to record (a usable snapshot that fell through has
        // one by construction).
        let needsListing = !requested.subtracting(defaults).isEmpty
        let listed = needsListing ? await listRepository(repoID, client: client, revision: revision) : nil
        try Task.checkCancellation()
        let listing = listed?.entries
        let fetchRevision = listed?.revision ?? revision

        let fetched: Set<String>
        if usableSnapshot {
            // The weights are here; fetch only the listed files the uncovered
            // patterns name that are not on disk, one by one into the same
            // directory, so nothing already present is downloaded, listed or
            // copied again. With no listing (offline) the snapshot is served
            // as it is, uncertified: the loader finds out whether the files it
            // wants are there, as it always did.
            let missing = uncoveredPatterns(requested, recorded: recorded, defaults: defaults, in: modelDir)
            guard let listing else {
                print("Using cached model at: \(modelDir.path) (the Hub could not be asked for \(missing.sorted().joined(separator: ", ")))")
                return modelDir
            }
            let wanted = listing.filter { entry in
                entry.type == .file && missing.contains { fnmatch($0, entry.path, 0) == 0 }
            }
            // A listing comes from the network: a path that leaves the model
            // directory (`tokenizer/../../x` matches `tokenizer*`, since `*`
            // matches `/`) is refused outright, as the snapshot download
            // refuses it, rather than written where it points.
            for entry in wanted {
                try Self.validateEntryPath(entry.path, under: modelDir)
            }
            let absent = wanted.filter { !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent($0.path).path) }
            if !absent.isEmpty {
                print("Fetching \(absent.count) file(s) for \(repoID): \(absent.map(\.path).sorted().joined(separator: ", "))...")
                for entry in absent {
                    _ = try await client.downloadFile(
                        entry,
                        from: repoID,
                        to: modelDir.appendingPathComponent(entry.path),
                        kind: .model,
                        revision: fetchRevision
                    )
                }
            }
            // Record only what is now on disk: a pattern whose listed files
            // still are not there (a fetch that returned without them) is
            // asked about again next time.
            let incomplete = wanted.filter { !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent($0.path).path) }
            fetched = missing.filter { pattern in
                !incomplete.contains { fnmatch(pattern, $0.path, 0) == 0 }
            }
            print("Model ready at: \(modelDir.path)")
        } else {
            let patterns = defaults.union(requested).union(recorded)
            let progress: @MainActor @Sendable (Progress) -> Void = progressHandler ?? { progress in
                print("\(progress.completedUnitCount)/\(progress.totalUnitCount) files")
            }
            print("Downloading model \(repoID)...")
            _ = try await client.downloadSnapshot(
                of: repoID,
                kind: .model,
                to: modelDir,
                revision: revision,
                matching: Array(patterns),
                progressHandler: progress
            )
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
            // Recorded from evidence, as above: the snapshot download serves
            // the client's own cached snapshot when it cannot list, which may
            // lack what the listing taken a moment earlier names, so only a
            // pattern whose listed files are all on disk now is complete.
            if let listing {
                let lacking = listing.filter { entry in
                    entry.type == .file
                        && requested.contains { fnmatch($0, entry.path, 0) == 0 }
                        && !FileManager.default.fileExists(atPath: modelDir.appendingPathComponent(entry.path).path)
                }
                fetched = requested.filter { pattern in
                    !lacking.contains { fnmatch(pattern, $0.path, 0) == 0 }
                }
            } else {
                fetched = []
            }
        }

        // Only a listing proves the patterns are complete; a download that the
        // client served from its own cache proves nothing about them, so a
        // fresh fetch records only what it saw land (`recorded` is empty on
        // that branch: a snapshot that was not usable had its manifest dropped
        // above, and a cleared one has none).
        if listing != nil {
            recordPatterns(modelDir: modelDir, patterns: recorded.union(fetched))
        }
        return modelDir
    }

    /// The Hub host, `HF_ENDPOINT` or huggingface.co: what `HubClient` picks
    /// on its own when no token is given, applied to the token client too so
    /// a mirror user is not sent to huggingface.co for one of the two.
    public static var hubHost: URL {
        // The same parse as `HubClient`'s, so both clients read one value
        // the same way.
        if let endpoint = ProcessInfo.processInfo.environment["HF_ENDPOINT"],
           let url = URL(string: endpoint) {
            return url
        }
        return HubClient.defaultHost
    }

    /// The requested patterns a snapshot is not known to be complete for: not
    /// among the patterns it was fetched with, not a default every fetch
    /// includes, and not a single-file name (no glob characters) whose file
    /// is on disk.
    private static func uncoveredPatterns(
        _ requested: Set<String>, recorded: Set<String>, defaults: Set<String>, in modelDir: URL
    ) -> Set<String> {
        requested.filter { pattern in
            if recorded.contains(pattern) || defaults.contains(pattern) { return false }
            let isLiteral = !pattern.contains { "*?[".contains($0) }
            if isLiteral, FileManager.default.fileExists(atPath: modelDir.appendingPathComponent(pattern).path) {
                return false
            }
            return true
        }
    }

    /// Repos whose listing ran out of time or took seconds to fail, with
    /// when. A network that drops packets rather than refusing them costs a
    /// cold load one attempt of at most `listingTimeout`, a proxy that
    /// answers late costs it those seconds, and the next cold loads of that
    /// repo nothing for `listingRetryInterval`, instead of that per resolve
    /// under the lock. A failure that comes back at once (no network, a
    /// refused connection, a definite answer such as a 404) costs nothing to
    /// repeat and is not remembered.
    private static let slowListingFailures = OSAllocatedUnfairLock<[String: Date]>(initialState: [:])
    static let listingTimeout: Duration = .seconds(10)
    static let slowFailureThreshold: Duration = .seconds(2)
    static let listingRetryInterval: TimeInterval = 600

    /// Test seam: forget which repos could not be listed in time.
    static func forgetSlowListingFailures() {
        slowListingFailures.withLock { $0.removeAll() }
    }

    /// A listing and the revision it was made at.
    struct Listing {
        let entries: [Git.TreeEntry]
        let revision: String
    }

    private enum ListingAttempt {
        case listed([Git.TreeEntry])
        case notFound
        case failed
        case timedOut
    }

    private static func attemptListing(
        _ repoID: Repo.ID, client: HubClient, revision: String
    ) async -> ListingAttempt {
        await withTaskGroup(of: ListingAttempt.self) { group in
            group.addTask {
                do {
                    return .listed(try await client.listFiles(in: repoID, kind: .model, revision: revision, recursive: true))
                } catch let error as HTTPClientError {
                    if case .responseError(let response, _) = error, response.statusCode == 404 {
                        return .notFound
                    }
                    return .failed
                } catch {
                    return .failed
                }
            }
            group.addTask {
                // Cancelled with the group once the listing answered, or with
                // the caller: neither is the clock running out.
                do {
                    try await Task.sleep(for: listingTimeout)
                    return .timedOut
                } catch {
                    return .failed
                }
            }
            let first = await group.next() ?? .failed
            group.cancelAll()
            return first
        }
    }

    private static func listRepository(
        _ repoID: Repo.ID, client: HubClient, revision: String
    ) async -> Listing? {
        let key = repoID.description
        if let failedAt = slowListingFailures.withLock({ $0[key] }),
           Date().timeIntervalSince(failedAt) < listingRetryInterval {
            return nil
        }
        // Measured over both attempts when the second is needed: every cold
        // load would repeat the pair, so its cost is the sum.
        let started = ContinuousClock.now
        var attempt = await attemptListing(repoID, client: client, revision: revision)
        var usedRevision = revision
        // A cached commit the Hub no longer has (a re-upload, a force-push):
        // the snapshot is completed from `main` instead of never.
        if case .notFound = attempt, revision != "main" {
            attempt = await attemptListing(repoID, client: client, revision: "main")
            usedRevision = "main"
        }
        // A caller that was cancelled got no answer; that is not the Hub's.
        if Task.isCancelled { return nil }
        switch attempt {
        case .listed(let entries):
            slowListingFailures.withLock { _ = $0.removeValue(forKey: key) }
            return Listing(entries: entries, revision: usedRevision)
        case .timedOut:
            print("Listing \(key) ran out of time (\(listingTimeout)); not asked again for \(Int(listingRetryInterval)) s")
            slowListingFailures.withLock { $0[key] = Date() }
            return nil
        case .failed, .notFound:
            // A failure that took seconds (a proxy answering late, a stalled
            // handshake, a 404 that a mirror takes its time over) would cost
            // every cold load those seconds under the lock, so it is
            // remembered like a timeout; one that came back at once costs
            // nothing to repeat.
            let elapsed = ContinuousClock.now - started
            if elapsed >= slowFailureThreshold {
                print("Listing \(key) failed after \(elapsed); not asked again for \(Int(listingRetryInterval)) s")
                slowListingFailures.withLock { $0[key] = Date() }
            }
            return nil
        }
    }

    /// The rule the snapshot download applies to every listed path, plus the
    /// check that the destination stays inside `modelDir`.
    static func validateEntryPath(_ path: String, under modelDir: URL) throws {
        guard !path.trimmingCharacters(in: .whitespaces).isEmpty,
              !path.contains("\0"), !path.contains("\\"), !path.hasPrefix("/"),
              path.split(separator: "/", omittingEmptySubsequences: false)
                  .allSatisfy({ !$0.isEmpty && $0 != ".." })
        else {
            throw ModelUtilsError.unsafeEntryPath(path)
        }
        // Lexical, like the cache's own path rule: the filesystem has no say
        // in whether a listed path stays under the directory.
        let root = modelDir.standardized.path
        let destination = modelDir.appendingPathComponent(path).standardized.path
        guard destination.hasPrefix(root + "/") else {
            throw ModelUtilsError.unsafeEntryPath(path)
        }
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

    private static func forgetRecordedPatterns(modelDir: URL) {
        try? FileManager.default.removeItem(at: modelDir.appendingPathComponent(patternsManifestName))
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
    /// The Hub listed a path that cannot be written under the model directory.
    case unsafeEntryPath(String)

    public var errorDescription: String? {
        switch self {
        case .incompleteDownload(let repo):
            return "Downloaded model '\(repo)' has missing or zero-byte weight files. "
                + "The cache has been cleared — please try again."
        case .unsafeEntryPath(let path):
            return "The repository listing names a path that cannot be written under the model directory: '\(path)'."
        }
    }
}
