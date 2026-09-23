enum TrackRopeCacheBounds {
    static func capacity(for indexerBudget: Int) -> Int? {
        guard indexerBudget > 0 else { return nil }
        return min(indexerBudget, 4096)
    }

    static func range(offset: Int, count: Int, capacity: Int) -> Range<Int>? {
        guard capacity > 0, offset >= 0, count > 0,
            offset <= capacity, count <= capacity - offset
        else { return nil }
        return offset ..< offset + count
    }
}
