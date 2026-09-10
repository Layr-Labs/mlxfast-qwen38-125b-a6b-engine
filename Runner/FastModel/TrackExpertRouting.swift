// Select experts and normalize their scores in one dispatch. Each SIMD
// group selects one token's experts; the threadgroup then schedules pairs
// by expert ID to improve weight-cache reuse across the verify window.
// Output slots stay in the original token/top-k order. Only execution order
// changes, and every expert dot product uses the existing arithmetic.
import MLX

enum TrackExpertRouting {
    static let source = #"""

const uint lane=thread_index_in_simdgroup;
const uint row=thread_position_in_threadgroup.y;
threadgroup uint selectedTable[ROWS * TOP];
constexpr int R=(E+31)/32;
float values[R]; uint indices[R];
for(int i=0;i<R;i++) { uint j=lane+32*i; values[i]=j<E?logits[row*E+j]:-INFINITY;indices[i]=j; }
float selected[TOP]; uint selectedIds[TOP];
for(int k=0;k<TOP;k++) {
    float best=-INFINITY; uint bestId=0xffffffff;
    for(int i=0;i<R;i++) {
        if(values[i]>best || (values[i]==best && indices[i]<bestId)) { best=values[i];bestId=indices[i]; }
    }
    float v=simd_max(best);
    uint id=simd_min(best==v?bestId:0xffffffff);
    selected[k]=v;selectedIds[k]=id;
    for(int i=0;i<R;i++) if(indices[i]==id) {values[i]=-INFINITY;indices[i]=0xffffffff;}
}
float local[4];float maximum=-INFINITY;
for(int i=0;i<4;i++){int k=lane*4+i;local[i]=k<TOP?selected[k]:-INFINITY;maximum=metal::max(maximum,local[i]);}
maximum=simd_max(maximum);maximum=simd_max(lane==0?maximum:-INFINITY);
float sum=0;
for(int i=0;i<4;i++){local[i]=fast::exp(local[i]-maximum);sum+=local[i];}
sum=simd_sum(sum);sum=simd_sum(lane==0?sum:0.0f);float inv=1.0f/sum;
for(int i=0;i<4;i++){int k=lane*4+i;if(k<TOP){idx[row*TOP+k]=selectedIds[k];weights[row*TOP+k]=local[i]*inv;}}

if (lane < TOP) selectedTable[row * TOP + lane] = selectedIds[lane];
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane < TOP) {
    const uint pair = row * TOP + lane;
    const uint expert = selectedIds[lane];
    uint rank = 0;
    for (uint j = 0; j < ROWS * TOP; ++j) {
        const uint other = selectedTable[j];
        rank += (other < expert || (other == expert && j < pair));
    }
    order[rank] = pair;
}
"""#

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_expert_select_order", inputNames: ["logits"],
        outputNames: ["idx", "weights", "order"], source: source)

    static func select(_ logits: MLXArray, topK: Int) -> [MLXArray] {
        let rows = logits.dim(1), experts = logits.dim(2)
        precondition(logits.dim(0) == 1 && rows >= 1 && rows <= 8)
        precondition(topK > 0 && topK <= 32 && topK <= experts && experts <= 512)
        precondition(logits.dtype == .float32)
        return kernel(
            [logits], template: [("E", experts), ("TOP", topK), ("ROWS", rows)],
            grid: (32, rows, 1), threadGroup: (32, rows, 1),
            outputShapes: [[1, rows, topK], [1, rows, topK], [rows * topK]],
            outputDTypes: [.uint32, .float32, .uint32])
    }
}
