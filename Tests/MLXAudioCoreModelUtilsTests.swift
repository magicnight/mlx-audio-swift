//  Tests for ModelUtils cache-completeness tracking (.mlx-audio-patterns).
//
//  Run this suite:
//    xcodebuild test \
//      -scheme MLXAudio-Package \
//      -destination 'platform=macOS' \
//      -parallel-testing-enabled NO \
//      -only-testing:'MLXAudioTests/ModelUtilsCacheTests' \
//      CODE_SIGNING_ALLOWED=NO

import Testing
import Foundation
import os
import HuggingFace
@testable import MLXAudioCore

extension Result {
    fileprivate var isFailure: Bool {
        if case .failure = self { return true }
        return false
    }
}

private func makeTemporaryCacheDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("modelutils-cache-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
}

/// Counts network requests per host and always fails with `.notConnectedToInternet`;
/// used to assert that a cache hit performs zero network traffic.
private final class OfflineCacheProtocol: URLProtocol, @unchecked Sendable {
    static let requests = OSAllocatedUnfairLock(initialState: [String: Int]())

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        Self.requests.withLock {
            $0[request.url!.host!, default: 0] += 1
        }
        client?.urlProtocol(
            self,
            didFailWithError: URLError(.notConnectedToInternet)
        )
    }

    override func stopLoading() {}
}

/// Answers the repo tree listing for hosts it knows, serves the files it was
/// given (HEAD and GET under `/resolve/`), and fails every other request,
/// counting listings and other requests per host: a Hub that can be asked
/// what a pattern names and hands out exactly the files a test allows.
private final class ListingProtocol: URLProtocol, @unchecked Sendable {
    static let listings = OSAllocatedUnfairLock(initialState: [String: Data]())
    static let served = OSAllocatedUnfairLock(initialState: [String: [String: Data]]())
    static let listingCounts = OSAllocatedUnfairLock(initialState: [String: Int]())
    static let listingPaths = OSAllocatedUnfairLock(initialState: [String: [String]]())
    static let otherRequests = OSAllocatedUnfairLock(initialState: [String: [String]]())
    /// Hosts whose requests are held open until the client gives up.
    static let hanging = OSAllocatedUnfairLock(initialState: Set<String>())
    /// Revisions a host answers a tree listing for with a 404, per host.
    static let notFoundRevisions = OSAllocatedUnfairLock(initialState: [String: Set<String>]())
    /// Per host, the number of listings answered before every further one
    /// fails with a 500.
    static let failAfterListings = OSAllocatedUnfairLock(initialState: [String: Int]())
    /// Hosts whose listings answer only after this many seconds: with an
    /// error, or with a 404 for a revision in `notFoundRevisions`.
    static let delayedFailures = OSAllocatedUnfairLock(initialState: [String: TimeInterval]())
    /// Set once `stopLoading` has reached this instance: a late answer is
    /// then dropped. An instance flag, since an object identifier can be
    /// reused once an earlier instance is gone.
    private let stopped = OSAllocatedUnfairLock(initialState: false)

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let host = url.host!
        if Self.hanging.withLock({ $0.contains(host) }) {
            Self.listingCounts.withLock { $0[host, default: 0] += 1 }
            return  // answered never; `stopLoading` ends it
        }
        if url.path.contains("/tree/"), let delay = Self.delayedFailures.withLock({ $0[host] }) {
            Self.listingCounts.withLock { $0[host, default: 0] += 1 }
            Self.listingPaths.withLock { $0[host, default: []].append(url.path) }
            let revision = url.path.components(separatedBy: "/tree/").last?
                .split(separator: "/").first.map(String.init) ?? ""
            let notFound = Self.notFoundRevisions.withLock { $0[host]?.contains(revision) ?? false }
            DispatchQueue.global().asyncAfter(deadline: .now() + delay) {
                if self.stopped.withLock({ $0 }) { return }
                if notFound {
                    let response = HTTPURLResponse(
                        url: url, statusCode: 404, httpVersion: "HTTP/1.1",
                        headerFields: ["Content-Type": "application/json"])!
                    self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                    self.client?.urlProtocolDidFinishLoading(self)
                } else {
                    self.client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
                }
            }
            return
        }
        if url.path.contains("/tree/"),
           let body = Self.listings.withLock({ $0[host] }) {
            let count: Int = Self.listingCounts.withLock { counts in
                counts[host, default: 0] += 1
                return counts[host, default: 0]
            }
            Self.listingPaths.withLock { $0[host, default: []].append(url.path) }
            // `/api/models/<ns>/<name>/tree/<revision>`.
            let revision = url.path.components(separatedBy: "/tree/").last?
                .split(separator: "/").first.map(String.init) ?? ""
            let status: Int
            if Self.notFoundRevisions.withLock({ $0[host]?.contains(revision) ?? false }) {
                status = 404
            } else if let limit = Self.failAfterListings.withLock({ $0[host] }), count > limit {
                status = 500
            } else {
                status = 200
            }
            let response = HTTPURLResponse(
                url: url, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            if status == 200 {
                client?.urlProtocol(self, didLoad: body)
            }
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        Self.otherRequests.withLock { $0[host, default: []].append("\(request.httpMethod ?? "?") \(url.path)") }
        if let range = url.path.range(of: "/resolve/") {
            // `/resolve/<revision>/<path>`: the file after the revision.
            let afterRevision = url.path[range.upperBound...].split(separator: "/", maxSplits: 1)
            if afterRevision.count == 2,
               let body = Self.served.withLock({ $0[host]?[String(afterRevision[1])] }) {
                let response = HTTPURLResponse(
                    url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                    headerFields: [
                        "ETag": "\"\(afterRevision[1])-etag\"",
                        "X-Repo-Commit": String(repeating: "a", count: 40),
                        "Content-Length": "\(body.count)",
                    ])!
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                if request.httpMethod != "HEAD" {
                    client?.urlProtocol(self, didLoad: body)
                }
                client?.urlProtocolDidFinishLoading(self)
                return
            }
        }
        client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
    }

    override func stopLoading() {
        stopped.withLock { $0 = true }
    }
}

private struct ListingFixture {
    let root: URL
    let cache: HubCache
    let repoID: Repo.ID
    let modelDir: URL
    let host: URL
    let session: URLSession

    static let commit = String(repeating: "a", count: 40)

    /// `files` is what the Hub lists for the repo, as (path, type) pairs;
    /// `served` the files it hands out, by path. `cachedCommit` writes the
    /// Hub cache's `refs/main` the way a first download would, so the resolve
    /// works at that commit; `populated` false leaves the model directory
    /// absent (a fresh download); `hangs` makes the Hub never answer;
    /// `notFoundRevisions` are answered 404 for a listing; after
    /// `failAfterListings` listings every further one is answered 500;
    /// `listingFailsAfter` makes every listing fail after that many seconds.
    init(
        files: [(String, String)], served: [String: Data] = [:],
        cachedCommit: Bool = false, populated: Bool = true, hangs: Bool = false,
        notFoundRevisions: Set<String> = [], failAfterListings: Int? = nil,
        listingFailsAfter: TimeInterval? = nil
    ) throws {
        ModelUtils.forgetSlowListingFailures()
        root = try makeTemporaryCacheDirectory()
        cache = HubCache(cacheDirectory: root)
        repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/listing-fixture"))
        if populated {
            modelDir = try populateCachedModel(cache: cache, repoID: repoID, patternsManifest: "")
        } else {
            modelDir = cache.cacheDirectory
                .appendingPathComponent("mlx-audio")
                .appendingPathComponent(repoID.description.replacingOccurrences(of: "/", with: "_"))
        }
        if cachedCommit {
            try cache.updateRef(repo: repoID, kind: .model, ref: "main", commit: Self.commit)
        }
        let listingHost = URL(string: "https://listing-\(UUID().uuidString.lowercased()).invalid")!
        host = listingHost
        if hangs {
            ListingProtocol.hanging.withLock { _ = $0.insert(listingHost.host!) }
        }
        let entries = files.map { ["type": $0.1, "path": $0.0, "oid": "0", "size": 1] as [String: Any] }
        let body = try JSONSerialization.data(withJSONObject: entries)
        let hostName = listingHost.host!
        ListingProtocol.listings.withLock { $0[hostName] = body }
        ListingProtocol.served.withLock { $0[hostName] = served }
        if !notFoundRevisions.isEmpty {
            ListingProtocol.notFoundRevisions.withLock { $0[hostName] = notFoundRevisions }
        }
        if let failAfterListings {
            ListingProtocol.failAfterListings.withLock { $0[hostName] = failAfterListings }
        }
        if let listingFailsAfter {
            ListingProtocol.delayedFailures.withLock { $0[hostName] = listingFailsAfter }
        }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ListingProtocol.self]
        session = URLSession(configuration: configuration)
    }

    var client: HubClient { HubClient(session: session, host: host, cache: cache) }

    var listingCount: Int {
        ListingProtocol.listingCounts.withLock { $0[host.host!] ?? 0 }
    }

    var listingPaths: [String] {
        ListingProtocol.listingPaths.withLock { $0[host.host!] ?? [] }
    }

    /// Every request that was not a listing, as "METHOD path".
    var otherRequests: [String] {
        ListingProtocol.otherRequests.withLock { $0[host.host!] ?? [] }
    }

    var otherRequestCount: Int { otherRequests.count }

    /// The identity of the weights file, which a resolve must never replace.
    var weightsIdentifier: AnyHashable? {
        let values = try? modelDir.appendingPathComponent("model.safetensors")
            .resourceValues(forKeys: [.fileResourceIdentifierKey])
        return values?.fileResourceIdentifier.flatMap { $0 as? NSObject }.map { AnyHashable($0) }
    }

    var manifest: Set<String> {
        let text = (try? String(
            contentsOf: modelDir.appendingPathComponent(".mlx-audio-patterns"), encoding: .utf8)) ?? ""
        return Set(text.split(separator: "\n").map(String.init))
    }

    func cleanUp() {
        session.invalidateAndCancel()
        ListingProtocol.listings.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.served.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.listingCounts.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.listingPaths.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.otherRequests.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.hanging.withLock { _ = $0.remove(host.host!) }
        ListingProtocol.notFoundRevisions.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.failAfterListings.withLock { _ = $0.removeValue(forKey: host.host!) }
        ListingProtocol.delayedFailures.withLock { _ = $0.removeValue(forKey: host.host!) }
        try? FileManager.default.removeItem(at: root)
    }
}

/// A client whose every request fails at once, counted under its own host:
/// for a test that expects a cache hit to make no request at all.
private func offlineClient(cache: HubCache) -> (client: HubClient, host: String, session: URLSession) {
    let host = URL(string: "https://offline-\(UUID().uuidString.lowercased()).invalid")!
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [OfflineCacheProtocol.self]
    let session = URLSession(configuration: configuration)
    return (HubClient(session: session, host: host, cache: cache), host.host!, session)
}

private struct OfflineCacheFixture {
    let root: URL
    let cache: HubCache
    let repoID: Repo.ID
    let modelDir: URL
    let host: URL
    let session: URLSession

    init() throws {
        ModelUtils.forgetSlowListingFailures()
        root = try makeTemporaryCacheDirectory()
        cache = HubCache(cacheDirectory: root)
        repoID = try #require(
            Repo.ID(rawValue: "mlx-audio-tests/cache-fixture")
        )
        // Pre-populate the mlx-audio layout an earlier download produced.
        modelDir = try populateCachedModel(
            cache: cache,
            repoID: repoID,
            patternsManifest: ""
        )

        // Reproduce the initial partial download in BOTH caches so the
        // snapshot lookup below sees the shape a real offline re-entry
        // produces: the mlx-audio layout under `modelDir`, and a hub
        // snapshot under <cache>/models/<ns>/<name>/snapshots/<hash>
        // containing just config.json and model.safetensors.
        let commit = String(repeating: "a", count: 40)
        let snapshot = try cache.snapshotPath(
            repo: repoID,
            kind: .model,
            commitHash: commit
        )
        try FileManager.default.createDirectory(
            at: snapshot,
            withIntermediateDirectories: true
        )
        for name in ["config.json", "model.safetensors"] {
            try FileManager.default.copyItem(
                at: modelDir.appendingPathComponent(name),
                to: snapshot.appendingPathComponent(name)
            )
        }
        try cache.updateRef(
            repo: repoID,
            kind: .model,
            ref: "main",
            commit: commit
        )

        // Unique hosts isolate request counts when tests run concurrently.
        host = URL(
            string: "https://cache-\(UUID().uuidString.lowercased()).invalid"
        )!
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [OfflineCacheProtocol.self]
        session = URLSession(configuration: configuration)
    }

    var client: HubClient {
        HubClient(session: session, host: host, cache: cache)
    }

    var requestCount: Int {
        OfflineCacheProtocol.requests.withLock { $0[host.host!] ?? 0 }
    }

    func cleanUp() {
        session.invalidateAndCancel()
        OfflineCacheProtocol.requests.withLock {
            _ = $0.removeValue(forKey: host.host!)
        }
        try? FileManager.default.removeItem(at: root)
    }
}

/// Pre-populates the on-disk cache layout that `resolveOrDownloadModel`
/// inspects: `<cache>/mlx-audio/<repo with _>/` containing a valid
/// config.json and a non-empty weights file.
private func populateCachedModel(
    cache: HubCache,
    repoID: Repo.ID,
    patternsManifest: String?
) throws -> URL {
    let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
    let modelDir = cache.cacheDirectory
        .appendingPathComponent("mlx-audio")
        .appendingPathComponent(modelSubdir)
    try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: modelDir.appendingPathComponent("config.json"))
    try Data([0x01]).write(to: modelDir.appendingPathComponent("model.safetensors"))
    if let patternsManifest {
        try Data(patternsManifest.utf8).write(
            to: modelDir.appendingPathComponent(".mlx-audio-patterns"))
    }
    return modelDir
}

@Suite("ModelUtils cache completeness", .serialized)
struct ModelUtilsCacheTests {

    @Test func cachedModelWithoutManifestServesPlainLoad() async throws {
        let cacheDir = try makeTemporaryCacheDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }
        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/cache-fixture"))
        let modelDir = try populateCachedModel(cache: cache, repoID: repoID, patternsManifest: nil)
        let offline = offlineClient(cache: cache)
        defer { offline.session.invalidateAndCancel() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: offline.client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors"
        )
        #expect(resolved.standardizedFileURL == modelDir.standardizedFileURL)
        #expect(OfflineCacheProtocol.requests.withLock { $0[offline.host] ?? 0 } == 0, "a hit makes no request")
    }

    @Test func cachedModelWithCoveringManifestServesPatternLoad() async throws {
        let cacheDir = try makeTemporaryCacheDirectory()
        defer { try? FileManager.default.removeItem(at: cacheDir) }
        let cache = HubCache(cacheDirectory: cacheDir)
        let repoID = try #require(Repo.ID(rawValue: "mlx-audio-tests/cache-fixture"))
        let modelDir = try populateCachedModel(
            cache: cache, repoID: repoID, patternsManifest: "*.mvn\n*.model")
        let offline = offlineClient(cache: cache)
        defer { offline.session.invalidateAndCancel() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: offline.client,
            cache: cache,
            repoID: repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == modelDir.standardizedFileURL)
        #expect(OfflineCacheProtocol.requests.withLock { $0[offline.host] ?? 0 } == 0, "a recorded pattern is a hit, with no request")
    }

    @Test func cachedModelMissingPatternsAsksTheHub() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }

        // A snapshot that lacks a requested pattern is not accepted as it is:
        // the Hub is asked what the pattern names. Here it cannot answer, so
        // the request is the whole observable effect.
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.requestCount > 0, "a missing pattern reaches the network")
    }

    @Test func offlinePartialSnapshotDoesNotCertifyMissingPatterns() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }

        // Without a listing the snapshot is served as it is, so this does not
        // throw; what matters is that the attempt reached the network and
        // that nothing certified "*.mvn".
        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.requestCount > 0)

        let manifest = try? String(
            contentsOf: fixture.modelDir
                .appendingPathComponent(".mlx-audio-patterns"),
            encoding: .utf8
        )
        #expect(
            !(manifest ?? "")
                .split(separator: "\n")
                .contains("*.mvn")
        )
    }

    @Test func offlineCertifiesNothingEvenForFilesOnDisk() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }
        try Data([0x02]).write(to: fixture.modelDir.appendingPathComponent("am.mvn"))

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn", "*.model"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)

        // Only the Hub's listing can say what "*.mvn" names; a file on disk
        // does not prove the pattern is complete, so the manifest stays as
        // the first fetch wrote it.
        let manifest = try String(
            contentsOf: fixture.modelDir.appendingPathComponent(".mlx-audio-patterns"),
            encoding: .utf8
        )
        #expect(manifest == "")
    }

    @Test func aPatternTheRepoHasNoFileForIsCompleteOnceTheHubSaysSo() async throws {
        let fixture = try ListingFixture(files: [("config.json", "file"), ("model.safetensors", "file")])
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.model"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.otherRequestCount == 0, "nothing to fetch, so nothing is downloaded or copied")
        #expect(fixture.manifest.contains("*.model"), "the listing proves the pattern complete")

        // From now on the snapshot is a hit: no request at all.
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.model"]
        )
        #expect(fixture.otherRequestCount == 0)
        #expect(fixture.listingCount == 1, "the Hub was asked once, for the first resolve only")
    }

    @Test func aSingleFileNameOnDiskNeedsNoListing() async throws {
        let fixture = try ListingFixture(files: [("config.json", "file"), ("model.safetensors", "file")])
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["config.json", "model.safetensors"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.listingCount == 0, "a name with no glob characters whose file is on disk is complete by itself")
        #expect(fixture.otherRequestCount == 0)
    }

    @Test func aListedFileMissingFromDiskIsFetchedAloneAndTheWeightsAreNotTouched() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            served: ["am.mvn": Data("mvn bytes".utf8)])
        defer { fixture.cleanUp() }
        let weightsBefore = try #require(fixture.weightsIdentifier)

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        let fetched = try Data(contentsOf: fixture.modelDir.appendingPathComponent("am.mvn"))
        #expect(fetched == Data("mvn bytes".utf8), "the listed file the pattern names is fetched")
        #expect(fixture.manifest.contains("*.mvn"))
        #expect(fixture.listingCount == 1)
        #expect(
            fixture.otherRequests.allSatisfy { $0.hasSuffix("/am.mvn") },
            "only the missing file is asked for: \(fixture.otherRequests)")
        #expect(fixture.weightsIdentifier == weightsBefore, "the weights already on disk are neither downloaded nor replaced")
    }

    @Test func theDifferenceIsFetchedAtTheCachedCommit() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            served: ["am.mvn": Data("mvn bytes".utf8)], cachedCommit: true)
        defer { fixture.cleanUp() }

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        let commit = ListingFixture.commit
        #expect(fixture.listingPaths.allSatisfy { $0.contains("/tree/\(commit)") }, "listed at the cached commit: \(fixture.listingPaths)")
        #expect(fixture.otherRequests.allSatisfy { $0.contains("/resolve/\(commit)/am.mvn") }, "fetched at the cached commit: \(fixture.otherRequests)")
        let refs = (try? FileManager.default.contentsOfDirectory(
            atPath: fixture.cache.refsDirectory(repo: fixture.repoID, kind: .model).path)) ?? []
        #expect(refs == ["main"], "no ref is written for the commit itself: \(refs)")
        #expect(fixture.manifest.contains("*.mvn"))
    }

    @Test func aListedPathThatLeavesTheModelDirectoryIsRefused() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("tokenizer/../../escape.txt", "file")],
            served: ["tokenizer/../../escape.txt": Data("escaped".utf8)])
        defer { fixture.cleanUp() }

        await #expect(throws: ModelUtilsError.self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: fixture.client,
                cache: fixture.cache,
                repoID: fixture.repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["tokenizer*"]
            )
        }
        #expect(fixture.otherRequestCount == 0, "nothing is fetched from a listing that names such a path")
        // `tokenizer/../../escape.txt` under the model directory points at
        // `<root>/mlx-audio/escape.txt`.
        #expect(!FileManager.default.fileExists(atPath: fixture.root.appendingPathComponent("mlx-audio/escape.txt").path))
        #expect(!FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("escape.txt").path))
        #expect(!fixture.manifest.contains("tokenizer*"))
    }

    @Test func theEntryPathRuleRefusesWhatCannotBeWrittenUnderTheModelDirectory() throws {
        let modelDir = URL(fileURLWithPath: "/tmp/model-dir", isDirectory: true)
        for path in ["/x", "a//b", "a\\b", "a\0b", "..", "a/../b", ".", "", " ", "a/"] {
            #expect(throws: ModelUtilsError.self, "\(path.debugDescription) must be refused") {
                try ModelUtils.validateEntryPath(path, under: modelDir)
            }
        }
        for path in ["a..b", "./a", "a/b.c", "tokenizer/vocab.json", ".mlx-audio-patterns"] {
            #expect(throws: Never.self, "\(path.debugDescription) must be allowed") {
                try ModelUtils.validateEntryPath(path, under: modelDir)
            }
        }
    }

    @Test func aCachedCommitTheHubNoLongerHasIsCompletedFromMain() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            served: ["am.mvn": Data("mvn bytes".utf8)], cachedCommit: true,
            notFoundRevisions: [ListingFixture.commit])
        defer { fixture.cleanUp() }

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        let revisions = fixture.listingPaths.map { $0.components(separatedBy: "/tree/").last ?? "" }
        #expect(revisions == [ListingFixture.commit, "main"], "the cached commit first, then main: \(fixture.listingPaths)")
        #expect(fixture.otherRequests.allSatisfy { $0.contains("/resolve/main/am.mvn") }, "fetched at main: \(fixture.otherRequests)")
        #expect(try Data(contentsOf: fixture.modelDir.appendingPathComponent("am.mvn")) == Data("mvn bytes".utf8))
        #expect(fixture.manifest.contains("*.mvn"))
    }

    @Test func aRepositoryTheHubNoLongerHasIsServedAsItIsAndAskedAgainNextTime() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file")], cachedCommit: true,
            notFoundRevisions: [ListingFixture.commit, "main"])
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL, "served as it is")
        #expect(fixture.listingCount == 2, "the cached commit, then main")
        #expect(fixture.otherRequestCount == 0)
        #expect(!fixture.manifest.contains("*.mvn"))
        // A definite answer costs nothing to ask for again: no backoff.
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 4, "asked again at once")
    }

    @Test func aListingTheHubRefusedOutrightBooksNoBackoff() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file")], failAfterListings: 0)
        defer { fixture.cleanUp() }

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 1)
        #expect(!fixture.manifest.contains("*.mvn"))
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "a quick failure is asked about again next time")
    }

    @Test func aFreshFetchServedFromTheClientsOwnCacheCertifiesNothingItLacks() async throws {
        // The client's cache already holds a `models--` snapshot of the config
        // and weights (the library's delete leaves exactly this behind, since
        // it removes the model directory only), and the Hub lists am.mvn too.
        // Our listing succeeds; the snapshot download's own listing then fails
        // and it serves the cached snapshot, without am.mvn.
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            cachedCommit: true, populated: false, failAfterListings: 1)
        defer { fixture.cleanUp() }
        let snapshot = try fixture.cache.snapshotPath(
            repo: fixture.repoID, kind: .model, commitHash: ListingFixture.commit)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try Data([0x01]).write(to: snapshot.appendingPathComponent("model.safetensors"))

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.listingCount == 2, "ours, then the snapshot download's")
        #expect(
            FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("model.safetensors").path),
            "served from the client's own cache")
        #expect(!FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("am.mvn").path))
        #expect(!fixture.manifest.contains("*.mvn"), "a file the download did not bring is not certified")
    }

    @Test func aFreshFetchOfTheDefaultsAloneDoesNotListForCertification() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file")],
            served: ["config.json": Data("{}".utf8), "model.safetensors": Data([0x01])],
            populated: false)
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors"
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.listingCount == 1, "only the snapshot download's own listing; nothing to certify")
        #expect(FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("model.safetensors").path))
    }

    @Test func aCancelledResolveLeavesNoBackoffBehind() async throws {
        let fixture = try ListingFixture(files: [("config.json", "file"), ("model.safetensors", "file")], hangs: true)
        defer { fixture.cleanUp() }

        let resolving = Task {
            try await ModelUtils.resolveOrDownloadModel(
                client: fixture.client,
                cache: fixture.cache,
                repoID: fixture.repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["*.mvn"]
            )
        }
        // Wait (polling) until the listing is in flight, then cancel the caller.
        let deadline = ContinuousClock.now + .seconds(30)
        while fixture.listingCount == 0, ContinuousClock.now < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        #expect(fixture.listingCount == 1)
        resolving.cancel()
        let outcome = await resolving.result
        #expect(outcome.isFailure, "a cancelled caller is not answered from the cache")

        // Not booked as the Hub's failure: the next resolve asks again.
        ListingProtocol.hanging.withLock { _ = $0.remove(fixture.host.host!) }
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "the Hub is asked again after a cancellation")
    }

    @Test func aListingThatRanOutOfTimeIsNotRetriedForAWhile() async throws {
        let fixture = try ListingFixture(files: [("config.json", "file"), ("model.safetensors", "file")], hangs: true)
        defer { fixture.cleanUp() }

        // The Hub never answers: the attempt gives up after `listingTimeout`
        // and the snapshot is served as it is.
        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.listingCount == 1)
        #expect(!fixture.manifest.contains("*.mvn"))
        ListingProtocol.hanging.withLock { _ = $0.remove(fixture.host.host!) }
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 1, "a repo whose listing ran out of time is not asked again at once")
        ModelUtils.forgetSlowListingFailures()
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "once the backoff is over it asks again, and this time the Hub answers")
        #expect(fixture.manifest.contains("*.mvn"))
    }

    @Test func aListingThatTakesSecondsToFailIsNotRetriedForAWhile() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file")], listingFailsAfter: 2.5)
        defer { fixture.cleanUp() }

        // A proxy that answers late: the failure costs those seconds under
        // the lock, so it is remembered like a timeout.
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 1)
        #expect(!fixture.manifest.contains("*.mvn"))
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 1, "a failure that took seconds is not asked about again at once")
        ModelUtils.forgetSlowListingFailures()
        ListingProtocol.delayedFailures.withLock { _ = $0.removeValue(forKey: fixture.host.host!) }
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "once the backoff is over it asks again")
        #expect(fixture.manifest.contains("*.mvn"))
    }

    @Test func aManifestBesideAnUnusableSnapshotIsDroppedBeforeAFreshFetch() async throws {
        // The half-state: weights and a manifest naming "*.mvn" survived, the
        // config did not. The Hub cannot be listed at all, and the snapshot
        // download falls back to a cached `models--` snapshot (config and
        // weights, no am.mvn). The inherited manifest must not turn the next
        // resolve into a hit on a snapshot that lacks the file.
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            cachedCommit: true, populated: false, failAfterListings: 0)
        defer { fixture.cleanUp() }
        try FileManager.default.createDirectory(at: fixture.modelDir, withIntermediateDirectories: true)
        try Data([0x01]).write(to: fixture.modelDir.appendingPathComponent("model.safetensors"))
        try Data("*.mvn".utf8).write(to: fixture.modelDir.appendingPathComponent(".mlx-audio-patterns"))
        let snapshot = try fixture.cache.snapshotPath(
            repo: fixture.repoID, kind: .model, commitHash: ListingFixture.commit)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: snapshot.appendingPathComponent("config.json"))
        try Data([0x01]).write(to: snapshot.appendingPathComponent("model.safetensors"))

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(FileManager.default.fileExists(atPath: fixture.modelDir.appendingPathComponent("config.json").path), "served from the client's cache")
        #expect(!fixture.manifest.contains("*.mvn"), "the inherited manifest is gone")
        let listingsSoFar = fixture.listingCount
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount > listingsSoFar, "the next resolve asks the Hub rather than trusting the old manifest")
        #expect(!fixture.manifest.contains("*.mvn"))
    }

    @Test func aManifestBesideAnUnusableSnapshotIsDroppedEvenWhenTheFetchFails() async throws {
        // The same half-state, with nothing in the client's cache to fall
        // back on: the fetch fails outright. The manifest must already be
        // gone, or a fetch that failed after copying the config would leave
        // it beside a snapshot the next resolve can use.
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            populated: false, failAfterListings: 0)
        defer { fixture.cleanUp() }
        try FileManager.default.createDirectory(at: fixture.modelDir, withIntermediateDirectories: true)
        try Data([0x01]).write(to: fixture.modelDir.appendingPathComponent("model.safetensors"))
        try Data("*.mvn".utf8).write(to: fixture.modelDir.appendingPathComponent(".mlx-audio-patterns"))

        await #expect(throws: (any Error).self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: fixture.client,
                cache: fixture.cache,
                repoID: fixture.repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["*.mvn"]
            )
        }
        #expect(!fixture.manifest.contains("*.mvn"), "dropped before the fetch, not after it")
    }

    @Test func aRepositoryTheHubNoLongerHasThatAnswersLateIsNotRetriedForAWhile() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file")], cachedCommit: true,
            notFoundRevisions: [ListingFixture.commit, "main"], listingFailsAfter: 1.5)
        defer { fixture.cleanUp() }

        // Two late 404s (the commit, then main): three seconds every cold load
        // would pay under the lock, so the pair is remembered like a timeout.
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2)
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "a 404 pair that took seconds is not asked about again at once")
    }

    @Test func aFreshFetchWithAPatternCertifiesItOnceItsFilesAreOnDisk() async throws {
        let fixture = try ListingFixture(
            files: [("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file")],
            served: [
                "config.json": Data("{}".utf8), "model.safetensors": Data([0x01]),
                "am.mvn": Data("mvn bytes".utf8),
            ],
            populated: false)
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(resolved.standardizedFileURL == fixture.modelDir.standardizedFileURL)
        #expect(fixture.listingCount == 2, "ours, to certify, then the snapshot download's own")
        #expect(try Data(contentsOf: fixture.modelDir.appendingPathComponent("am.mvn")) == Data("mvn bytes".utf8))
        #expect(fixture.manifest.contains("*.mvn"), "the file the pattern names landed, so the pattern is complete")
        let requestsAfterFetch = fixture.otherRequestCount
        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.listingCount == 2, "a hit from now on")
        #expect(fixture.otherRequestCount == requestsAfterFetch)
    }

    @Test func aListedFileAlreadyOnDiskIsNotFetchedAgain() async throws {
        let fixture = try ListingFixture(files: [
            ("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file"),
        ])
        defer { fixture.cleanUp() }
        try Data([0x02]).write(to: fixture.modelDir.appendingPathComponent("am.mvn"))

        _ = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.mvn"]
        )
        #expect(fixture.otherRequestCount == 0, "the file is there; only the listing was needed")
        #expect(fixture.manifest.contains("*.mvn"))
    }

    @Test func aListedFileMissingFromDiskIsFetchedAndNothingIsCertifiedWhenThatFails() async throws {
        let fixture = try ListingFixture(files: [
            ("config.json", "file"), ("model.safetensors", "file"), ("am.mvn", "file"),
        ])
        defer { fixture.cleanUp() }

        // The listing names am.mvn and the disk lacks it: the difference is
        // fetched. This Hub serves no files, so the fetch fails, and the
        // pattern must not be recorded as complete.
        await #expect(throws: (any Error).self) {
            _ = try await ModelUtils.resolveOrDownloadModel(
                client: fixture.client,
                cache: fixture.cache,
                repoID: fixture.repoID,
                requiredExtension: "safetensors",
                additionalMatchingPatterns: ["*.mvn"]
            )
        }
        #expect(fixture.otherRequestCount > 0, "the missing file was asked for")
        #expect(!fixture.manifest.contains("*.mvn"))
    }

    @Test func defaultPatternDoesNotRedownloadCachedSnapshot() async throws {
        let fixture = try OfflineCacheFixture()
        defer { fixture.cleanUp() }

        let resolved = try await ModelUtils.resolveOrDownloadModel(
            client: fixture.client,
            cache: fixture.cache,
            repoID: fixture.repoID,
            requiredExtension: "safetensors",
            additionalMatchingPatterns: ["*.json"]
        )

        #expect(
            resolved.standardizedFileURL
                == fixture.modelDir.standardizedFileURL
        )
        #expect(fixture.requestCount == 0)
    }
}
