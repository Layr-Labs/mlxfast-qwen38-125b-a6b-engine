import Foundation

enum TrackGDNTemporalPrep {
    static func source(_ original: String) -> String {
        func replace(_ source: String, _ old: String, _ new: String) -> String {
            precondition(source.components(separatedBy: old).count == 2)
            return source.replacingOccurrences(of: old, with: new)
        }
        var source = replace(
            original,
            """
            const uint bt = thread_position_in_grid.z;            // b*T + t
            const uint b = bt / T;
            const uint t = bt % T;
            """,
            """
            const uint tile = thread_position_in_grid.z;
            const uint b = tile / ((T + 1) / 2);
            const uint first_t = (tile % ((T + 1) / 2)) * 2;
            """)
        source = replace(
            source, "float thread_x[N_READS];",
            """
            float window[KC + 1][N_READS];
            InT weights[KC][N_READS];
            for (int i = 0; i < N_READS; ++i) {
                const uint ch = vec * 128 + lane * N_READS + i;
                for (int j = 0; j < KC + 1; ++j) {
                    const uint r = first_t + j;
                    window[j][i] = r < T + KM1 ? win((int)r, ch) : 0.0f;
                }
                for (int j = 0; j < KC; ++j) {
                    weights[j][i] = conv_w[ch * KC + j];
                }
            }
            for (int phase = 0; phase < 2; ++phase) {
                const uint t = first_t + phase;
                if (t >= T) { break; }
                const uint bt = b * T + t;
            float thread_x[N_READS];
            """)
        source = replace(
            source,
            "cacc += win((int)t + j, ch) * conv_w[ch * KC + j];",
            "cacc += window[phase + j][i] * weights[j][i];")
        return source + "\n}"
    }
}
