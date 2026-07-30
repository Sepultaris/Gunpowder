#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <bit>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace gunpowder {

// Sparse, spatially-addressed storage for a large two-dimensional cell world.
// Chunks are allocated lazily and remain contiguous internally so simulation
// passes can work on cache-friendly 64x64 neighborhoods.
template <typename T, int ChunkSize = 64>
class ChunkGrid {
public:
    static_assert(ChunkSize > 0);
    static_assert(std::has_single_bit(
        static_cast<unsigned int>(ChunkSize)));

    static constexpr int chunkSize = ChunkSize;
    static constexpr int chunkCellCount = ChunkSize * ChunkSize;

    // Temporary compatibility names while World migrates from SparseGrid.
    static constexpr int pageSize = chunkSize;
    static constexpr int pageCellCount = chunkCellCount;
    using Chunk = std::array<T, chunkCellCount>;
    using Page = Chunk;

    ChunkGrid() = default;
    ChunkGrid(int width, int height, T defaultValue = {})
        : width_(width),
          height_(height),
          chunksWide_((width + ChunkSize - 1) / ChunkSize),
          chunksHigh_((height + ChunkSize - 1) / ChunkSize),
          powerOfTwoWidth_(width > 0 && std::has_single_bit(
              static_cast<unsigned int>(width))),
          widthShift_(powerOfTwoWidth_
              ? static_cast<int>(std::countr_zero(
                    static_cast<unsigned int>(width)))
              : 0),
          widthMask_(powerOfTwoWidth_
              ? static_cast<std::size_t>(width - 1)
              : 0),
          defaultValue_(std::move(defaultValue)),
          chunks_(static_cast<std::size_t>(chunksWide_) *
                  static_cast<std::size_t>(chunksHigh_)) {
        assert(width >= 0);
        assert(height >= 0);
    }

    ChunkGrid(const ChunkGrid& other)
        : width_(other.width_),
          height_(other.height_),
          chunksWide_(other.chunksWide_),
          chunksHigh_(other.chunksHigh_),
          powerOfTwoWidth_(other.powerOfTwoWidth_),
          widthShift_(other.widthShift_),
          widthMask_(other.widthMask_),
          defaultValue_(other.defaultValue_),
          chunks_(other.chunks_.size()) {
        for (std::size_t key = 0; key < other.chunks_.size(); ++key) {
            if (other.chunks_[key]) {
                chunks_[key] =
                    std::make_unique<Chunk>(*other.chunks_[key]);
            }
        }
    }

    ChunkGrid& operator=(const ChunkGrid& other) {
        if (this == &other) {
            return *this;
        }
        ChunkGrid copy(other);
        *this = std::move(copy);
        return *this;
    }

    ChunkGrid(ChunkGrid&&) noexcept = default;
    ChunkGrid& operator=(ChunkGrid&&) noexcept = default;

    [[nodiscard]] int width() const { return width_; }
    [[nodiscard]] int height() const { return height_; }
    [[nodiscard]] int chunksWide() const { return chunksWide_; }
    [[nodiscard]] int chunksHigh() const { return chunksHigh_; }
    [[nodiscard]] std::size_t size() const {
        return static_cast<std::size_t>(width_) *
               static_cast<std::size_t>(height_);
    }
    [[nodiscard]] bool empty() const { return size() == 0; }

    [[nodiscard]] const T& operator[](std::size_t index) const {
        const Address address = addressOf(index);
        const auto& chunk = chunks_[address.key];
        return chunk ? (*chunk)[address.offset] : defaultValue_;
    }

    T& operator[](std::size_t index) {
        const Address address = addressOf(index);
        return ensureChunk(address.key)[address.offset];
    }

    [[nodiscard]] const T& get(int x, int y) const {
        return (*this)[linearIndex(x, y)];
    }

    void set(std::size_t index, const T& value) {
        const Address address = addressOf(index);
        auto& chunk = chunks_[address.key];
        if (!chunk) {
            if (value == defaultValue_) {
                return;
            }
            chunk = makeDefaultChunk();
        }
        (*chunk)[address.offset] = value;
    }

    void set(int x, int y, const T& value) {
        set(linearIndex(x, y), value);
    }

    void reset(T defaultValue = {}) {
        for (auto& chunk : chunks_) {
            chunk.reset();
        }
        defaultValue_ = std::move(defaultValue);
    }

    [[nodiscard]] bool hasChunk(int chunkX, int chunkY) const {
        if (chunkX < 0 || chunkY < 0 ||
            chunkX >= chunksWide_ || chunkY >= chunksHigh_) {
            return false;
        }
        return chunks_[chunkKey(chunkX, chunkY)] != nullptr;
    }

    // Allocate sparse storage before a parallel pass. Workers may then safely
    // write distinct cells without racing the lazy-allocation path.
    void ensureChunkAllocated(int chunkX, int chunkY) {
        assert(chunkX >= 0 && chunkX < chunksWide_);
        assert(chunkY >= 0 && chunkY < chunksHigh_);
        (void)ensureChunk(chunkKey(chunkX, chunkY));
    }

    [[nodiscard]] bool hasPage(int pageX, int pageY) const {
        return hasChunk(pageX, pageY);
    }

    [[nodiscard]] std::size_t allocatedChunkCount() const {
        return static_cast<std::size_t>(std::count_if(
            chunks_.begin(), chunks_.end(),
            [](const auto& chunk) { return chunk != nullptr; }));
    }

    template <typename Function>
    void forEachAllocatedChunk(Function&& function) const {
        for (int chunkY = 0; chunkY < chunksHigh_; ++chunkY) {
            for (int chunkX = 0; chunkX < chunksWide_; ++chunkX) {
                const auto& chunk =
                    chunks_[chunkKey(chunkX, chunkY)];
                if (chunk) {
                    function(chunkX, chunkY, *chunk);
                }
            }
        }
    }

    void releaseDefaultChunksOutside(int firstCellX, int firstCellY,
                                     int lastCellX, int lastCellY) {
        for (int chunkY = 0; chunkY < chunksHigh_; ++chunkY) {
            for (int chunkX = 0; chunkX < chunksWide_; ++chunkX) {
                auto& chunk = chunks_[chunkKey(chunkX, chunkY)];
                if (!chunk) {
                    continue;
                }

                const int chunkMinX = chunkX * ChunkSize;
                const int chunkMinY = chunkY * ChunkSize;
                const int chunkMaxX =
                    std::min(width_, chunkMinX + ChunkSize);
                const int chunkMaxY =
                    std::min(height_, chunkMinY + ChunkSize);
                const bool intersectsKeptArea =
                    chunkMinX < lastCellX && chunkMaxX > firstCellX &&
                    chunkMinY < lastCellY && chunkMaxY > firstCellY;
                if (intersectsKeptArea) {
                    continue;
                }

                if (std::all_of(
                        chunk->begin(), chunk->end(),
                        [&](const T& value) {
                            return value == defaultValue_;
                        })) {
                    chunk.reset();
                }
            }
        }
    }

    void releaseDefaultPagesOutside(int firstCellX, int firstCellY,
                                    int lastCellX, int lastCellY) {
        releaseDefaultChunksOutside(
            firstCellX, firstCellY, lastCellX, lastCellY);
    }

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator() = default;
        const_iterator(const ChunkGrid* grid, std::size_t index)
            : grid_(grid), index_(index) {}
        reference operator*() const { return (*grid_)[index_]; }
        pointer operator->() const { return &(*grid_)[index_]; }
        const_iterator& operator++() {
            ++index_;
            return *this;
        }
        const_iterator operator++(int) {
            const_iterator result = *this;
            ++(*this);
            return result;
        }
        friend bool operator==(const const_iterator& first,
                               const const_iterator& second) {
            return first.grid_ == second.grid_ &&
                   first.index_ == second.index_;
        }

    private:
        const ChunkGrid* grid_ = nullptr;
        std::size_t index_ = 0;
    };

    [[nodiscard]] const_iterator begin() const {
        return const_iterator(this, 0);
    }
    [[nodiscard]] const_iterator end() const {
        return const_iterator(this, size());
    }

    friend bool operator==(const ChunkGrid& first,
                           const ChunkGrid& second) {
        if (first.width_ != second.width_ ||
            first.height_ != second.height_) {
            return false;
        }
        for (std::size_t index = 0; index < first.size(); ++index) {
            if (first[index] != second[index]) {
                return false;
            }
        }
        return true;
    }

private:
    struct Address {
        std::size_t key = 0;
        std::size_t offset = 0;
    };

    [[nodiscard]] std::size_t linearIndex(int x, int y) const {
        assert(x >= 0 && x < width_);
        assert(y >= 0 && y < height_);
        return static_cast<std::size_t>(y) *
                   static_cast<std::size_t>(width_) +
               static_cast<std::size_t>(x);
    }

    [[nodiscard]] std::size_t chunkKey(int chunkX, int chunkY) const {
        return static_cast<std::size_t>(chunkY) *
                   static_cast<std::size_t>(chunksWide_) +
               static_cast<std::size_t>(chunkX);
    }

    [[nodiscard]] Address addressOf(std::size_t index) const {
        assert(index < size());
        std::size_t x = 0;
        std::size_t y = 0;
        if (powerOfTwoWidth_) {
            x = index & widthMask_;
            y = index >> widthShift_;
        } else {
            const std::size_t gridWidth =
                static_cast<std::size_t>(width_);
            x = index % gridWidth;
            y = index / gridWidth;
        }

        static constexpr int chunkShift =
            static_cast<int>(std::countr_zero(
                static_cast<unsigned int>(ChunkSize)));
        static constexpr std::size_t chunkMask =
            static_cast<std::size_t>(ChunkSize - 1);
        const int chunkX = static_cast<int>(x >> chunkShift);
        const int chunkY = static_cast<int>(y >> chunkShift);
        const std::size_t localX = x & chunkMask;
        const std::size_t localY = y & chunkMask;
        return {
            chunkKey(chunkX, chunkY),
            (localY << chunkShift) + localX,
        };
    }

    [[nodiscard]] std::unique_ptr<Chunk> makeDefaultChunk() const {
        auto chunk = std::make_unique<Chunk>();
        chunk->fill(defaultValue_);
        return chunk;
    }

    [[nodiscard]] Chunk& ensureChunk(std::size_t key) {
        auto& chunk = chunks_[key];
        if (!chunk) {
            chunk = makeDefaultChunk();
        }
        return *chunk;
    }

    int width_ = 0;
    int height_ = 0;
    int chunksWide_ = 0;
    int chunksHigh_ = 0;
    bool powerOfTwoWidth_ = false;
    int widthShift_ = 0;
    std::size_t widthMask_ = 0;
    T defaultValue_{};
    std::vector<std::unique_ptr<Chunk>> chunks_;
};

template <typename T, int PageSize = 64>
using SparseGrid = ChunkGrid<T, PageSize>;

} // namespace gunpowder
