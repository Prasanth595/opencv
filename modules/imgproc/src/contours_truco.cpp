
#include "precomp.hpp"
#include "contours_common.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <map>
#include <atomic>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <memory>
#if defined(_MSC_VER)
#include <intrin.h>   // __iso_volatile_load8 / __iso_volatile_store8
#endif

namespace{

// Tunable block size. 1024 points = 8KB (Fits easily in L1 Cache)
template <size_t BLOCK_SIZE = 2048>
class TRUCOPagedContour {
public:
    struct Block {
        cv::Point data[BLOCK_SIZE];
    };

    TRUCOPagedContour() {
        allocateBlock();
        // Initialize pointers to the start of the first block
        curr_ptr_ = all_blocks_[0]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    ~TRUCOPagedContour() {
        for (Block* b : all_blocks_) cv::fastFree(b);
    }

    // --- HOT PATH: Minimal instructions ---
    // No counter updates, just raw pointer arithmetic.
    inline void push_back(const cv::Point& pt) {
        if (curr_ptr_ == end_ptr_) {
            current_block_idx_++;
            if (current_block_idx_ == all_blocks_.size()) {
                allocateBlock();
            }
            curr_ptr_ = all_blocks_[current_block_idx_]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
        *curr_ptr_++ = pt;
    }

    inline void pop_back() {
        // Safety check: do nothing if completely empty
        if (current_block_idx_ == 0 && curr_ptr_ == all_blocks_[0]->data) return;

        // Check if we are at the start of the current block
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            // Move to the previous block
            current_block_idx_--;
            // Point to the end of the previous block
            curr_ptr_ = all_blocks_[current_block_idx_]->data + BLOCK_SIZE;
            end_ptr_  = curr_ptr_;
        }
        curr_ptr_--;
    }

    inline const cv::Point& back() const {
        // Handle case where back() crosses block boundary
        if (curr_ptr_ == all_blocks_[current_block_idx_]->data) {
            return all_blocks_[current_block_idx_ - 1]->data[BLOCK_SIZE - 1];
        }
        return *(curr_ptr_ - 1);
    }

    inline const cv::Point& front() const {
        return all_blocks_[0]->data[0];
    }

    // Calculated on demand (O(1) arithmetic, but slightly more math than reading a variable)
    size_t size() const {
        size_t elements_in_last = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        return (current_block_idx_ * BLOCK_SIZE) + elements_in_last;
    }

    void clear() {
        current_block_idx_ = 0;
        if (!all_blocks_.empty()) {
            curr_ptr_ = all_blocks_[0]->data;
            end_ptr_  = curr_ptr_ + BLOCK_SIZE;
        }
    }

    // Optimized Copy: Uses block-wise memcpy
    void copyTo(std::vector<cv::Point>& out) const {
        size_t total = size();
        out.resize(total);
        if (total == 0) return;

        cv::Point* dst = out.data();

        // 1. Copy full blocks
        for (size_t i = 0; i < current_block_idx_; ++i) {
            std::memcpy(dst, all_blocks_[i]->data, BLOCK_SIZE * sizeof(cv::Point));
            dst += BLOCK_SIZE;
        }

        // 2. Copy partial last block
        size_t last_block_count = curr_ptr_ - all_blocks_[current_block_idx_]->data;
        if (last_block_count > 0) {
            std::memcpy(dst, all_blocks_[current_block_idx_]->data, last_block_count * sizeof(cv::Point));
        }
    }

private:
    void grow() {
        current_block_idx_++;
        if (current_block_idx_ == all_blocks_.size()) {
            allocateBlock();
        }
        curr_ptr_ = all_blocks_[current_block_idx_]->data;
        end_ptr_  = curr_ptr_ + BLOCK_SIZE;
    }

    void allocateBlock() {
        Block* b = (Block*)cv::fastMalloc(sizeof(Block));
        all_blocks_.push_back(b);
    }

    std::vector<Block*> all_blocks_;
    size_t current_block_idx_ = 0;

    // Fast pointers for the hot loop
    cv::Point* curr_ptr_ = nullptr;
    cv::Point* end_ptr_  = nullptr;
};


////IMPLEMENTATION

// Relaxed atomic byte access to the shared label board (the padded image). The only bytes accessed
// by more than one thread are read by the other threads just to tell background from non-background,
// which never changes, so no ordering is needed: relaxed atomics make the algorithm data-race-free
// and compile to plain byte loads/stores. std::atomic_ref is C++20, hence the compiler-specific
// equivalents (also used by the standard libraries themselves to implement relaxed atomics).
static inline uchar loadRelaxed(const uchar* p)
{
#if defined(__cpp_lib_atomic_ref)
    return std::atomic_ref<uchar>(*const_cast<uchar*>(p)).load(std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
    return __atomic_load_n(p, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
    return (uchar)__iso_volatile_load8((const volatile char*)p);
#else
    return *(const volatile uchar*)p;
#endif
}
static inline void storeRelaxed(uchar* p, uchar v)
{
#if defined(__cpp_lib_atomic_ref)
    std::atomic_ref<uchar>(*p).store(v, std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(p, v, __ATOMIC_RELAXED);
#elif defined(_MSC_VER)
    __iso_volatile_store8((volatile char*)p, (char)v);
#else
    *(volatile uchar*)p = v;
#endif
}

// Per-contour data needed to build the hierarchy (only filled when the hierarchy is requested).
// The hierarchy of Suzuki-Abe is the containment of 1- and 0-components, so for every contour C
// it is enough to know ONE neighbouring border D found on the row where C starts:
//  - external C, started at the run start a of row r: D is the border through the end b of the
//    previous run of row r, on its EAST side (the background between b and a is a single
//    0-component). No previous run: C is a root.
//  - internal C, started at the run end b of row r: D is the border through the start a of the
//    same run, on its WEST side.
// Then parent(C) = (type(D) != type(C)) ? D : parent(D), as in the table of Suzuki-Abe.
struct ContourInfo {
    int64_t order;   // Suzuki-Abe order: start pixel (linear index in the source image) * 2 + isHole
    int64_t refPix;  // linear index of the pixel of D, -1: none (root); 64-bit: images of more than 2^31 pixels
    uchar refSide;   // REF_WEST or REF_EAST
    bool isHole;
};
enum : uchar { REF_WEST = 1, REF_EAST = 2 };

struct AccumulatorT:public std::vector<std::vector<cv::Point>>{
    std::vector<int> idx_internal_lastLine,idx_external_firstLine;
    std::vector<ContourInfo> info;
};

// The image is split in horizontal stripes [s_k, e_k), one per task. Correctness does not depend
// on the schedule of the tasks (the output is a pure function of the image) because of two rules:
//  1. Task k writes VISITED / VISITED_OUTER_RIGHT marks only on its own rows [s_k, e_k) of the shared
//     board, plus a private copy of row e_k (the first row of the next stripe, scanned by task k for
//     internal contours only). Traces still run freely downwards, but do not write outside those rows,
//     and background pixels are never written. Hence every decision of task k (the external/internal
//     start tests) reads only marks written by task k itself, while the other tasks only test
//     background vs. non-background on its rows.
//  2. Ownership is geometric. Let minrow(C) be the top row of contour C. An external contour is
//     accepted only from an attempt on row minrow(C), an internal one only from an attempt on row
//     minrow(C)+1; both attempts are the Suzuki-Abe border following start points of C. External
//     attempts are made on rows [s_k, e_k) and internal ones on rows (s_k, e_k], so each contour is
//     accepted exactly once, by the task whose own rows contain minrow(C). A trace that moves above
//     s_k is aborted.
class TRUCOntourTracer : public cv::ParallelLoopBody
{
public:

    // Accumulator: Vector of (Vector of Contours), where Contour is Vector of Points
    using AccumulatorType = std::vector<AccumulatorT>;

    // img   : padded binary image (0 / 255), the shared label board
    // extra : one private row per stripe (copy of row e_k), extra.row(k)
    // withInfo : also record the ContourInfo of every contour (hierarchy requested)
    TRUCOntourTracer(cv::Mat& img,
                     const std::vector<cv::Range>& stripRanges,
                     AccumulatorType& accumulator,
                     cv::Mat& extra,
                     size_t minSize,
                     bool withInfo = false)
        : padded_(img), extra_(extra), ranges_(stripRanges), accumulator_(accumulator), minSize_(minSize),
          withInfo_(withInfo)
    {
        step_ = padded_.step;
        int istep = (int)step_;
        // 0: East (Right)
        offsets_[0] = 1;
        // 1: NE (Up-Right)
        offsets_[1] = -istep + 1;
        // 2: North (Up)
        offsets_[2] = -istep;
        // 3: NW (Up-Left)
        offsets_[3] = -istep - 1;
        // 4: West (Left)
        offsets_[4] = -1;
        // 5: SW (Down-Left)
        offsets_[5] = istep - 1;
        // 6: South (Down)
        offsets_[6] = istep;
        // 7: SE (Down-Right)
        offsets_[7] = istep + 1;

        memcpy(offsets_ + 8, offsets_, 8 * sizeof(int));
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        // Pre-allocate buffer to avoid re-allocation during moves
        TRUCOPagedContour<4096> buffer;

        const int cols = padded_.cols;

        for (int k = range.start; k < range.end; ++k)
        {
            const int s = ranges_[k].start, e = ranges_[k].end;   // scans rows [s, e]
            auto& local_contours = accumulator_[k];

            // Hint for result vector size
            local_contours.reserve(2048);
            if (withInfo_)
                local_contours.info.reserve(2048);

            // private copy of row e, normalized to FOREGROUND / BACKGROUND
            // (the owner of row e may be writing marks on it concurrently)
            uchar* ext = extra_.data + (size_t)k * extra_.step;
            const uchar* src = padded_.data + (size_t)e * step_;
            for (int x = 0; x < cols; ++x)
                ext[x] = loadRelaxed(src + x) ? FOREGROUND : BACKGROUND;
            const Labels labels = { padded_.data, ext, step_, e };

            for (int r = s; r <= e; ++r)
            {
                uchar* row_ptr = (r < e) ? padded_.data + (size_t)r * step_ : ext;
                // linear index in the source image of the padded column x of row r: rowBase + x
                const int64_t rowBase = (int64_t)(r - 1) * (cols - 2) - 1;
                int prevEnd = -1;   // padded column of the end of the previous run of this row

                // "c" is updated by the find* functions
                for (int c = 1; c < cols - 1; )
                {
                    // 1. FAST SCAN: Skip background pixels
                    if ((c = findStartContourPoint(row_ptr, cols, c)) == cols) break;
                    const int runStart = c;

                    // 2. External contour: start point not visited yet, rows [s, e)
                    if (r < e && row_ptr[c] == FOREGROUND)
                    {
                        const ContourInfo info = { (rowBase + c) * 2,
                                                   prevEnd < 0 ? -1 : rowBase + prevEnd, REF_EAST, false };
                        if (traceContour(&buffer, r, c, s, e, labels, true) &&
                            emit(buffer, local_contours, info) && r == s)
                            // register this so we can zip them as Suzuki&Abe method
                            local_contours.idx_external_firstLine.push_back((int)(local_contours.size() - 1));
                    }

                    // 3. FAST SCAN: Find end of current component to skip processing it again
                    c = findEndContourPoint(row_ptr, cols, c + 1);
                    if (c >= cols) break; // end of row
                    prevEnd = c - 1;

                    // 4. Internal contour: rows (s, e]; its start point always has a non-background NE
                    //    neighbour (the pixel above the first background pixel of the hole), on row r-1,
                    //    which is always an own row
                    if (r > s && row_ptr[c - 1] != VISITED_OUTER_RIGHT &&
                        padded_.data[(size_t)(r - 1) * step_ + c] != BACKGROUND)
                    {
                        const ContourInfo info = { (rowBase + c - 1) * 2 + 1, rowBase + runStart, REF_WEST, true };
                        if (traceContour(&buffer, r, c - 1, s, e, labels, false) &&
                            emit(buffer, local_contours, info) && r == e)
                            // register this so we can zip them as Suzuki&Abe method
                            local_contours.idx_internal_lastLine.push_back((int)(local_contours.size() - 1));
                    }
                }
            }
        }
    }

    static inline int findStartContourPoint(const uchar* src_data, int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        cv::v_uint8 v_zero = cv::vx_setzero_u8();
        for (; j <= width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes())
        {
            cv::v_uint8 vmask = (cv::v_ne(cv::vx_load(src_data + j), v_zero));
            if (cv::v_check_any(vmask))
            {
                j += cv::v_scan_forward(vmask);
                return j;
            }
        }
#endif
        for (; j < width && !src_data[j]; ++j)
            ;
        return j;
    }

    inline static int findEndContourPoint(const uchar* src_data,int width, int j)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (j <  width && !src_data[j])
        {
            return j;
        }
        else
        {
            cv::v_uint8 v_zero = cv::vx_setzero_u8();
            for (; j <=  width - cv::VTraits<cv::v_uint8>::vlanes(); j += cv::VTraits<cv::v_uint8>::vlanes())
            {
                cv::v_uint8 vmask = (cv::v_eq(cv::vx_load(src_data + j), v_zero));
                if (cv::v_check_any(vmask))
                {
                    j += cv::v_scan_forward(vmask);
                    return j;
                }
            }
        }
#endif
        for (; j < width && src_data[j]; ++j)
            ;

        return j;
    }

private:
    // where the marks of a pixel (given by its index in the padded image and its row) live for a task
    struct Labels {
        uchar* shared; uchar* ext; size_t step; int e;
        inline uchar* at(size_t idx, int row) const {     // only valid for row <= e
            return row < e ? shared + idx : ext + (idx - (size_t)e * step);
        }
    };

    // adds the traced contour to the output; returns false if it is discarded by minSize
    inline bool emit(TRUCOPagedContour<4096>& buffer, AccumulatorT& local_contours, const ContourInfo& info) const
    {
        if (buffer.size() > 1 && buffer.back() == buffer.front())
            buffer.pop_back();
        if (buffer.size() < minSize_)
            return false;
        local_contours.emplace_back();
        buffer.copyTo(local_contours.back());
        if (withInfo_)
            local_contours.info.push_back(info);
        return true;
    }

    // Moore tracing from (r, c). Returns true iff the contour was traced completely AND this attempt
    // is its owner (external: started on its top row; internal: started one row below its top row).
    bool traceContour(TRUCOPagedContour<4096>* buffer, int r, int c, int s, int e,
                      const Labels& labels, bool isExternal) const
    {
        buffer->clear();
        const uchar* board = padded_.data;   // only background / non-background is read here
        int curr_x = c, curr_y = r, minrow = r;
        int start_dir = -1, dir = -1;
        int search_idx = isExternal ? 5 : 1;
        size_t curr = (size_t)r * step_ + c;
        const size_t start = curr;
        bool is_first_move = true;

        // TRACING LOOP
        while (true)
        {
            buffer->push_back({ (curr_x - 1), (curr_y - 1) });
            // Check neighbors
            for (int n = 0; n < 8; ++n)
            {
                int idx = search_idx + n;
                // Use offset cache
                const size_t neighbor = curr + offsets_[idx];
                if (loadRelaxed(board + neighbor) == BACKGROUND) continue;

                dir = idx & 7;
                // --- EXECUTE MOVE ---
                curr_y += dy_[dir];
                curr_x += dx_[dir];
                if (curr_y < s)
                    return false;   // moved into the stripe above: not ours
                if (curr_y < minrow)
                    minrow = curr_y;

                // mark the pixel we are leaving; only rows <= e are stored by this task
                const int prev_y = curr_y - dy_[dir];
                if (prev_y <= e)
                {
                    uchar* label = labels.at(curr, prev_y);
                    if ((search_idx <= 1) || (dir <= search_idx - 2))
                        storeRelaxed(label, VISITED_OUTER_RIGHT);
                    else if (*label == FOREGROUND)
                        storeRelaxed(label, VISITED_);
                }

                // Short-circuit Jacob's Check
                if (curr == start && !is_first_move && dir == start_dir)
                    return isExternal ? (minrow == r) : (minrow == r - 1);

                curr = neighbor; // move
                // Reset search index for Moore neighbor
                search_idx = (dir + 6) & 7;
                break;
            }
            if (is_first_move) {
                if (dir == -1) { // single pixel
                    storeRelaxed(labels.at(curr, r), VISITED_OUTER_RIGHT);
                    return isExternal;   // an isolated pixel is an external contour only
                }
                start_dir = dir;
                is_first_move = false;
            }
        }
    }

    cv::Mat& padded_;
    cv::Mat& extra_;
    const std::vector<cv::Range>& ranges_;
    AccumulatorType& accumulator_;
    size_t minSize_;
    bool withInfo_;
    size_t step_;
    int offsets_[16];

    // 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE (CCW Rotation)
    const int dx_[8] = {  1,  1,  0, -1, -1, -1,  0,  1 };
    const int dy_[8] = {  0, -1, -1, -1,  0,  1,  1,  1 };
    // Constants defined once
    enum : uchar { FOREGROUND = 255, BACKGROUND = 0, VISITED_OUTER_RIGHT = 100, VISITED_ = 200 };
};


void approxContour(std::vector<cv::Point> &inout,cv::ContourApproximationModes contApprox_)   {
    size_t n = inout.size();
    if (n <= 1) return;

    if (contApprox_ == cv::CHAIN_APPROX_SIMPLE) {
        std::vector<cv::Point> result;
        result.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            size_t prev_i = (i == 0) ? n - 1 : i - 1;
            size_t next_i = (i == n - 1) ? 0 : i + 1;

            cv::Point v1 = inout[i] - inout[prev_i];
            cv::Point v2 = inout[next_i] - inout[i];

            if (v1 != v2) {
                result.push_back(inout[i]);
            }
        }
        if (result.empty() && n > 0) result.push_back(inout[0]);
        inout = std::move(result);
    } else if (contApprox_ == cv::CHAIN_APPROX_TC89_L1 || contApprox_ == cv::CHAIN_APPROX_TC89_KCOS) {
        auto getCode = [](cv::Point d) -> schar {
            if (d.x == 1) {
                if (d.y == 0) return 0;
                if (d.y == -1) return 1;
                if (d.y == 1) return 7;
            } else if (d.x == 0) {
                if (d.y == -1) return 2;
                if (d.y == 1) return 6;
            } else if (d.x == -1) {
                if (d.y == -1) return 3;
                if (d.y == 0) return 4;
                if (d.y == 1) return 5;
            }
            return 0;
        };

        cv::ContourCodesStorage::storage_t codesStorage;
        cv::ContourCodesStorage codes(&codesStorage);
        for (size_t i = 0; i < n; ++i) {
            cv::Point delta = inout[(i + 1) % n] - inout[i];
            codes.push_back(getCode(delta));
        }

        cv::ContourPointsStorage::storage_t pointsStorage;
        cv::ContourPointsStorage points(&pointsStorage);
        cv::approximateChainTC89(codes, inout[0], contApprox_, points);

        inout.clear();
        inout.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            inout.push_back(points.at(i));
        }
    }
}

// ==========================================================
// 1. The Core Implementation (Operates on std::vector directly)
// ==========================================================
void findTRUContoursImpl(cv::Mat& padded,
                           std::vector<std::vector<cv::Point>>& outContours,
                           int minSize,int contApprox)
{
    //   Load Balancing Logic: one stripe of rows per thread (at least one row per stripe)
    const int innerRows = padded.rows - 2;
    const int nstripes = std::max(1, std::min(cv::getNumThreads(), innerRows));
    std::vector<cv::Range> balancedRanges;
    int rowsPerStripe = innerRows / nstripes;
    int remainingRows = innerRows % nstripes;
    int currentRow = 1;
    for (int t = 0; t < nstripes; ++t) {
        int startRow = currentRow;
        int endRow = startRow + rowsPerStripe + (t < remainingRows ? 1 : 0);
        balancedRanges.emplace_back(startRow, endRow);
        currentRow = endRow;
    }
    //  Parallel Execution
    std::vector<AccumulatorT> threadAccumulators(balancedRanges.size());
    cv::Mat extraRows((int)balancedRanges.size(), padded.cols, CV_8UC1);   // one private row per stripe
    TRUCOntourTracer worker(padded, balancedRanges, threadAccumulators, extraRows, minSize);
    cv::parallel_for_(cv::Range(0, (int)balancedRanges.size()), worker);


    // REORG To Match Suzuki & Abe's Contour Ordering.
    // Every adjacent strip pair (t, t+1) shares a boundary row. On that row:
    //   * thread t   recorded INTERNAL fragments -> idx_internal_lastLine (tail of accT)
    //   * thread t+1 recorded EXTERNAL fragments -> idx_external_firstLine (head of accT1)
    // Both runs are already in X-ascending order (left-to-right raster scan),
    // so a two-pointer merge restores the sequential Suzuki & Abe ordering.
    for (size_t t = 0; t + 1 < threadAccumulators.size(); ++t) {
        auto& accT  = threadAccumulators[t];
        auto& accT1 = threadAccumulators[t + 1];

        const size_t kI = accT.idx_internal_lastLine.size();
        const size_t kE = accT1.idx_external_firstLine.size();
        if (kI == 0 && kE == 0) continue;

        const size_t tailStart = accT.size() - kI;

        // Merge by ascending X of each contour's first point. Contours are moved,
        // not copied — only std::vector handles (three pointers) change hands.
        std::vector<std::vector<cv::Point>> merged;
        merged.reserve(kI + kE);
        size_t i = tailStart, iEnd = accT.size();
        size_t j = 0,         jEnd = kE;
        while (i < iEnd && j < jEnd) {
            if (accT[i].front().x <= accT1[j].front().x)
                merged.emplace_back(std::move(accT[i++]));
            else
                merged.emplace_back(std::move(accT1[j++]));
        }
        while (i < iEnd) merged.emplace_back(std::move(accT[i++]));
        while (j < jEnd) merged.emplace_back(std::move(accT1[j++]));

        // Replace accT's tail with the merged run.
        accT.resize(tailStart);
        for (auto& c : merged) accT.emplace_back(std::move(c));

        // Drop the consumed externals from the head of accT1.
        accT1.erase(accT1.begin(), accT1.begin() + kE);

        // Any remaining bookkeeping in accT1 indexed absolute positions — fix them.
        for (auto& idx : accT1.idx_internal_lastLine)
            idx -= static_cast<int>(kE);

        // This boundary's bookkeeping is now consumed.
        accT.idx_internal_lastLine.clear();
        accT1.idx_external_firstLine.clear();
    }

    //   ZERO-COPY MERGE
    size_t totalContours = 0;
    for (auto& tVec : threadAccumulators)
        totalContours += tVec.size();


    outContours.clear();
    outContours.reserve(totalContours);
    // move the contours from thread accumulators to output without copying pixel data
    for (auto& tVec : threadAccumulators) {
        // move_iterator moves the vector internals (pointers) without copying pixel data
        outContours.insert(outContours.end(),
                           std::make_move_iterator(tVec.begin()),
                           std::make_move_iterator(tVec.end()));
    }
    //  reverse the order to match original findContours Suzuki&Abe
    std::reverse(outContours.begin(), outContours.end());


    //7. now, lets do the contour approximation if needed in parallel
    if(contApprox!=cv::CHAIN_APPROX_NONE){
        cv::parallel_for_(cv::Range(0, (int)outContours.size()), [&](const cv::Range& range){
            for(int i=range.start;i<range.end;i++){
                approxContour(outContours[i],(cv::ContourApproximationModes)contApprox);
            }
        });
    }

}

// ==========================================================
// Hierarchy (RETR_LIST / RETR_CCOMP / RETR_TREE)
// ==========================================================

// direction (0=E, 1=NE, ..., 7=SE, y down) of the move a -> b between 8-neighbours
static inline int moveDir(const cv::Point& a, const cv::Point& b)
{
    static const int lut[3][3] = { { 3, 2, 1 }, { 4, -1, 0 }, { 5, 6, 7 } };   // [dy+1][dx+1]
    return lut[b.y - a.y + 1][b.x - a.x + 1];
}

// Background neighbours (REF_WEST / REF_EAST) examined by the border following at a point where
// the search started at direction sc and found the next point in direction d: the cyclic range
// [sc-1, d) (sc-1 is known to be background, it is the first pixel examined by Suzuki-Abe)
static inline uchar examinedSides(int sc, int d)
{
    const int from = (sc + 7) & 7;
    const int len = ((d - sc) & 7) + 1;
    uchar sides = 0;
    if (((4 - from) & 7) < len) sides |= REF_WEST;
    if (((0 - from) & 7) < len) sides |= REF_EAST;
    return sides;
}

// Sorts pairs (key, index) whose key is (pixel * 2 + bit) and that come from the concatenation of
// the stripe lists: each list is sorted, and two consecutive lists only overlap on the row they
// share, so only the windows of those rows need to be sorted.
static void sortStripeConcat(std::vector<std::pair<int64_t, int>>& v, int width)
{
    auto row = [&](size_t i) { return (v[i].first >> 1) / width; };
    const size_t n = v.size();
    for (size_t i = 1; i < n; ++i)
    {
        if (!(v[i] < v[i - 1])) continue;
        const int64_t r = row(i);
        size_t l = i, h = i;
        while (l > 0 && row(l - 1) == r) --l;
        while (h < n && row(h) == r) ++h;
        std::sort(v.begin() + l, v.begin() + h);
        i = h;
    }
}

// Finds, for every contour, the contour D its reference (pixel, side) belongs to. A pair (pixel,
// side whose neighbour is background) lies on exactly one border, so each reference is answered
// by exactly one contour: the contours are walked (in parallel, in chunks of points), the sides
// examined at each point are recomputed and looked up in a mask of referenced pixels.
class ReferenceResolver : public cv::ParallelLoopBody
{
public:
    enum { CHUNK = 1 << 15 };   // points per task

    ReferenceResolver(const std::vector<std::vector<cv::Point>>& contours, const std::vector<ContourInfo>& info,
                      int width, const uchar* mask, const std::vector<std::pair<int64_t, int>>& refs,
                      const std::vector<size_t>& firstPoint, std::vector<int>& target)
        : contours_(contours), info_(info), width_(width), mask_(mask), refs_(refs),
          firstPoint_(firstPoint), target_(target)
    {
        for (int din = 0; din < 8; ++din)
            for (int dout = 0; dout < 8; ++dout)
                sides_[din][dout] = examinedSides((din + 6) & 7, dout);
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        const size_t p0 = (size_t)range.start * CHUNK;
        const size_t p1 = std::min((size_t)range.end * CHUNK, firstPoint_.back());
        // contour that contains the point p0
        size_t i = std::upper_bound(firstPoint_.begin(), firstPoint_.end(), p0) - firstPoint_.begin() - 1;
        for (size_t p = p0; p < p1; ++i)
        {
            const std::vector<cv::Point>& c = contours_[i];
            const size_t m = c.size();
            const size_t j0 = p - firstPoint_[i], j1 = std::min(m, j0 + (p1 - p));
            p += j1 - j0;
            if (m == 0) continue;
            if (m == 1)   // isolated pixel
            {
                answer(c[0], REF_WEST | REF_EAST, (int)i);
                continue;
            }
            if (j0 == 0)   // first visit of the start point (search started at SW for external, NE for internal)
                answer(c[0], examinedSides(info_[i].isHole ? 1 : 5, moveDir(c[0], c[1])), (int)i);
            // every visit, including the closing one of the start point
            int din = moveDir(c[j0 == 0 ? m - 1 : j0 - 1], c[j0]);
            for (size_t j = j0; j < j1; ++j)
            {
                const int dout = moveDir(c[j], c[j + 1 == m ? 0 : j + 1]);
                answer(c[j], sides_[din][dout], (int)i);
                din = dout;
            }
        }
    }

private:
    inline void answer(const cv::Point& pt, uchar sides, int contour) const
    {
        const int64_t pix = (int64_t)pt.y * width_ + pt.x;
        const uchar hit = mask_[pix] & sides;
        if (!hit) return;
        for (int side = REF_WEST; side <= REF_EAST; ++side)
        {
            if (!(hit & side)) continue;
            const int64_t key = pix * 2 + (side - 1);
            auto it = std::lower_bound(refs_.begin(), refs_.end(), std::make_pair(key, INT_MIN));
            // a contour may visit a pixel several times, possibly in two chunks: same value
            storeRelaxedInt(&target_[it->second], contour);
        }
    }
    static inline void storeRelaxedInt(int* p, int v)
    {
#if defined(__cpp_lib_atomic_ref)
        std::atomic_ref<int>(*p).store(v, std::memory_order_relaxed);
#elif defined(__GNUC__) || defined(__clang__)
        __atomic_store_n(p, v, __ATOMIC_RELAXED);
#else
        *(volatile int*)p = v;
#endif
    }

    const std::vector<std::vector<cv::Point>>& contours_;
    const std::vector<ContourInfo>& info_;
    const int width_;
    const uchar* mask_;
    const std::vector<std::pair<int64_t, int>>& refs_;
    const std::vector<size_t>& firstPoint_;
    std::vector<int>& target_;
    uchar sides_[8][8];
};

// Contours and hierarchy laid out as cv::findContours does: every contour is added, in Suzuki-Abe
// order, as the FIRST child of its parent (CTree::addChild), and the output is the pre-order
// traversal of the tree (contourTreeToResults).
void findTRUContoursHierarchyImpl(cv::Mat& padded,
                                  std::vector<std::vector<cv::Point>>& outContours,
                                  std::vector<cv::Vec4i>& outHierarchy,
                                  int mode, int contApprox)
{
    const int width = padded.cols - 2, height = padded.rows - 2;
    const int nstripes = std::max(1, std::min(cv::getNumThreads(), height));
    std::vector<cv::Range> ranges;
    for (int t = 0, cur = 1; t < nstripes; ++t)
    {
        const int end = cur + height / nstripes + (t < height % nstripes ? 1 : 0);
        ranges.emplace_back(cur, end);
        cur = end;
    }

    // 1. tracing (parallel)
    std::vector<AccumulatorT> acc(ranges.size());
    cv::Mat extraRows((int)ranges.size(), padded.cols, CV_8UC1);
    TRUCOntourTracer worker(padded, ranges, acc, extraRows, 0, true);
    cv::parallel_for_(cv::Range(0, (int)ranges.size()), worker);

    size_t total = 0;
    for (auto& a : acc) total += a.size();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<ContourInfo> info;
    contours.reserve(total);
    info.reserve(total);
    for (auto& a : acc)
    {
        contours.insert(contours.end(), std::make_move_iterator(a.begin()), std::make_move_iterator(a.end()));
        info.insert(info.end(), a.info.begin(), a.info.end());
    }
    acc.clear();
    const int n = (int)total;

    // 2. references (parallel walk)
    std::vector<int> ref(n, -1);
    if (mode != cv::RETR_LIST)
    {
        std::unique_ptr<uchar, void (*)(void*)> mask((uchar*)std::calloc((size_t)width * height, 1), &std::free);
        CV_Assert(mask);
        std::vector<std::pair<int64_t, int>> refs;   // (pixel * 2 + (side == REF_EAST), contour)
        refs.reserve(n);
        std::vector<size_t> firstPoint(n + 1, 0);
        for (int i = 0; i < n; ++i)
        {
            firstPoint[i + 1] = firstPoint[i] + contours[i].size();
            if (info[i].refPix < 0) continue;
            mask.get()[info[i].refPix] |= info[i].refSide;
            refs.emplace_back(info[i].refPix * 2 + (info[i].refSide - 1), i);
        }
        sortStripeConcat(refs, width);
        const int nchunks = (int)((firstPoint[n] + ReferenceResolver::CHUNK - 1) / ReferenceResolver::CHUNK);
        ReferenceResolver resolver(contours, info, width, mask.get(), refs, firstPoint, ref);
        cv::parallel_for_(cv::Range(0, nchunks), resolver);
    }

    // 3. parents (RETR_TREE semantics). ref[i] precedes i in raster order, but not necessarily in
    //    stripe order, so the chains are resolved with an explicit stack
    std::vector<int> parent(n, -2);   // -2: unresolved, -1: root
    {
        std::vector<int> stack;
        for (int i = 0; i < n; ++i)
        {
            int j = i;
            while (parent[j] == -2)
            {
                const int d = ref[j];
                if (d < 0) { parent[j] = -1; break; }
                if (info[d].isHole != info[j].isHole) { parent[j] = d; break; }
                stack.push_back(j);
                j = d;
            }
            while (!stack.empty())
            {
                const int k = stack.back(); stack.pop_back();
                parent[k] = parent[ref[k]];
            }
        }
    }

    // 4. tree of the requested mode, children added in Suzuki-Abe order as first child
    std::vector<std::pair<int64_t, int>> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = { info[i].order, i };
    sortStripeConcat(sorted, width);
    std::vector<int> firstChild(n + 1, -1), next(n, -1), treeParent(n, -1);   // n: root
    for (int k = 0; k < n; ++k)
    {
        const int i = sorted[k].second;
        int p = parent[i];
        if (mode == cv::RETR_LIST || (mode == cv::RETR_CCOMP && !info[i].isHole))
            p = -1;
        treeParent[i] = p;
        const int pp = p < 0 ? n : p;
        next[i] = firstChild[pp];
        firstChild[pp] = i;
    }

    // 5. pre-order traversal
    std::vector<int> newIdx(n, -1), preorder;
    preorder.reserve(n);
    {
        std::vector<int> stack;
        auto pushChildren = [&](int node) {
            const size_t mark = stack.size();
            for (int c = firstChild[node]; c >= 0; c = next[c]) stack.push_back(c);
            std::reverse(stack.begin() + mark, stack.end());
        };
        pushChildren(n);
        while (!stack.empty())
        {
            const int i = stack.back(); stack.pop_back();
            newIdx[i] = (int)preorder.size();
            preorder.push_back(i);
            pushChildren(i);
        }
    }

    const int kept = (int)preorder.size();
    outContours.clear();
    outContours.reserve(kept);
    outHierarchy.assign(kept, cv::Vec4i(-1, -1, -1, -1));
    for (int ni = 0; ni < kept; ++ni)
    {
        const int i = preorder[ni];
        cv::Vec4i& h = outHierarchy[ni];
        h[0] = next[i] < 0 ? -1 : newIdx[next[i]];
        if (h[0] >= 0) outHierarchy[h[0]][1] = ni;
        h[2] = firstChild[i] < 0 ? -1 : newIdx[firstChild[i]];
        h[3] = treeParent[i] < 0 ? -1 : newIdx[treeParent[i]];
        outContours.push_back(std::move(contours[i]));
    }

    // 6. contour approximation if needed, in parallel
    if (contApprox != cv::CHAIN_APPROX_NONE)
    {
        cv::parallel_for_(cv::Range(0, kept), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++)
                approxContour(outContours[i], (cv::ContourApproximationModes)contApprox);
        });
    }
}
}

namespace cv{

// ==========================================================
//
//  Public API: Handles OutputArray and dispatches to the core implementation
//  This is a modified version of the original TRUCO parallel algorithm to produce the exact same output
//  as original findContours with RETR_LIST mode (no hierarchy, all contours are external). It also supports contour approximation.
//
// ==========================================================
void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, int minSize, bool binarize,int method)
{
    CV_INSTRUMENT_REGION();
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);

    // Buffer handling
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, 0);   // ISOLATED: ignore the pixels outside a ROI
    if (binarize)
        cv::threshold(padded, padded, 0, 255, cv::THRESH_BINARY);


    // Fast path: caller passed std::vector<std::vector<cv::Point>> directly.
    // Write into it without any intermediate copy.
    if (_contours.kind() == _InputArray::STD_VECTOR_VECTOR) {
        auto* vec = reinterpret_cast<std::vector<std::vector<cv::Point>>*>(_contours.getObj());
        findTRUContoursImpl(padded, *vec, minSize,method);
    }
    else{ // Slow path: generic OutputArray — build in a temp vector then copy.
        std::vector<std::vector<cv::Point>> tempContours;
        findTRUContoursImpl(padded, tempContours, minSize,method);

        _contours.create((int)tempContours.size(), 1, 0, -1, true);
        for (size_t i = 0; i < tempContours.size(); i++) {
            _contours.create((int)tempContours[i].size(), 1, CV_32SC2, (int)i, true);
            Mat m = _contours.getMat((int)i);
            std::memcpy(m.data, tempContours[i].data(), tempContours[i].size() * sizeof(cv::Point));
        }
    }
}

// ==========================================================
//
//  Same contours as findTRUContours, with the hierarchy and the contour order of findContours for
//  mode RETR_LIST, RETR_CCOMP or RETR_TREE. The hierarchy is written only if
//  _hierarchy is needed.
//
// ==========================================================
void findTRUContours(InputArray _src, OutputArrayOfArrays _contours, OutputArray _hierarchy,
                     int mode, int method, bool binarize)
{
    CV_INSTRUMENT_REGION();
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(mode == RETR_LIST || mode == RETR_CCOMP || mode == RETR_TREE);

    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, 0);   // ISOLATED: ignore the pixels outside a ROI
    if (binarize)
        cv::threshold(padded, padded, 0, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> tempContours;
    std::vector<std::vector<cv::Point>>* contours = &tempContours;
    if (_contours.kind() == _InputArray::STD_VECTOR_VECTOR)
        contours = reinterpret_cast<std::vector<std::vector<cv::Point>>*>(_contours.getObj());
    std::vector<Vec4i> hierarchy;
    findTRUContoursHierarchyImpl(padded, *contours, hierarchy, mode, method);

    if (contours->empty())
    {
        _contours.clear();
        return;
    }
    if (contours == &tempContours)
    {
        _contours.create((int)tempContours.size(), 1, 0, -1, true);
        for (size_t i = 0; i < tempContours.size(); i++)
        {
            _contours.create((int)tempContours[i].size(), 1, CV_32SC2, (int)i, true);
            Mat m = _contours.getMat((int)i);
            std::memcpy(m.data, tempContours[i].data(), tempContours[i].size() * sizeof(cv::Point));
        }
    }
    if (_hierarchy.needed())
    {
        _hierarchy.create(1, (int)hierarchy.size(), CV_32SC4, -1, true);
        std::memcpy(_hierarchy.getMat().data, hierarchy.data(), hierarchy.size() * sizeof(Vec4i));
    }
}

}
