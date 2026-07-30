#pragma once

#include "game/math.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <random>
#include <utility>
#include <vector>

namespace gunpowder {

class VulkanRenderer;

struct MaterialSimulationTimings {
    float totalMs = 0.0F;
    float granularMs = 0.0F;
    float liquidPreparationMs = 0.0F;
    float liquidTransportMs = 0.0F;
    float liquidSelectionMs = 0.0F;
    float liquidGravityMs = 0.0F;
    float liquidLateralMs = 0.0F;
    float liquidFrontierMs = 0.0F;
    float liquidEqualizationMs = 0.0F;
    float gasAndReactionMs = 0.0F;
    float heatMs = 0.0F;
    std::uint32_t activeChunks = 0;
    std::uint32_t liquidCandidateVisits = 0;
    std::uint32_t equalizedComponents = 0;
    std::uint32_t equalizedCells = 0;
    bool valid = false;
};

template <typename T, int PageSize = 64>
class SparseGrid {
public:
    static constexpr int pageSize = PageSize;
    static constexpr int pageCellCount = PageSize * PageSize;
    using Page = std::array<T, pageCellCount>;

    SparseGrid() = default;
    SparseGrid(int width, int height, T defaultValue = {})
        : width_(width),
          height_(height),
          defaultValue_(defaultValue),
          pages_(static_cast<std::size_t>(
              (static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height) +
               pageCellCount - 1) /
              pageCellCount)) {}

    SparseGrid(const SparseGrid& other)
        : width_(other.width_),
          height_(other.height_),
          defaultValue_(other.defaultValue_),
          pages_(other.pages_.size()) {
        for (std::size_t key = 0; key < other.pages_.size(); ++key) {
            if (other.pages_[key]) {
                pages_[key] =
                    std::make_unique<Page>(*other.pages_[key]);
            }
        }
    }
    SparseGrid& operator=(const SparseGrid& other) {
        if (this == &other) {
            return *this;
        }
        SparseGrid copy(other);
        *this = std::move(copy);
        return *this;
    }
    SparseGrid(SparseGrid&&) noexcept = default;
    SparseGrid& operator=(SparseGrid&&) noexcept = default;

    [[nodiscard]] std::size_t size() const {
        return static_cast<std::size_t>(width_) *
               static_cast<std::size_t>(height_);
    }
    [[nodiscard]] bool empty() const { return size() == 0; }

    [[nodiscard]] const T& operator[](std::size_t index) const {
        const auto [key, offset] = pageAddress(index);
        return pages_[key] ? (*pages_[key])[offset] : defaultValue_;
    }
    T& operator[](std::size_t index) {
        const auto [key, offset] = pageAddress(index);
        auto& page = pages_[key];
        if (!page) {
            page = std::make_unique<Page>();
            page->fill(defaultValue_);
        }
        return (*page)[offset];
    }

    void set(std::size_t index, const T& value) {
        const auto [key, offset] = pageAddress(index);
        if (!pages_[key]) {
            if (value == defaultValue_) {
                return;
            }
            auto page = std::make_unique<Page>();
            page->fill(defaultValue_);
            (*page)[offset] = value;
            pages_[key] = std::move(page);
            return;
        }
        (*pages_[key])[offset] = value;
    }

    void reset(T defaultValue = {}) {
        for (auto& page : pages_) {
            page.reset();
        }
        defaultValue_ = defaultValue;
    }

    [[nodiscard]] bool hasPage(int pageX, int pageY) const {
        const std::size_t firstIndex =
            static_cast<std::size_t>(pageY * PageSize) *
                static_cast<std::size_t>(width_) +
            static_cast<std::size_t>(pageX * PageSize);
        const std::size_t key =
            firstIndex / static_cast<std::size_t>(pageCellCount);
        return key < pages_.size() && pages_[key] != nullptr;
    }

    template <typename Function>
    void forEachAllocatedPage(Function&& function) const {
        for (std::size_t key = 0; key < pages_.size(); ++key) {
            if (!pages_[key]) {
                continue;
            }
            function(key * static_cast<std::size_t>(pageCellCount),
                     *pages_[key]);
        }
    }

    void releaseDefaultPagesOutside(int firstCellX, int firstCellY,
                                    int lastCellX, int lastCellY) {
        for (std::size_t key = 0; key < pages_.size(); ++key) {
            auto& page = pages_[key];
            if (!page) {
                continue;
            }
            const std::size_t pageBegin =
                key * static_cast<std::size_t>(pageCellCount);
            const std::size_t pageEnd =
                std::min(size(), pageBegin +
                                     static_cast<std::size_t>(
                                         pageCellCount));
            bool intersectsKeptArea = false;
            std::size_t cursor = pageBegin;
            while (cursor < pageEnd) {
                const int y = static_cast<int>(
                    cursor / static_cast<std::size_t>(width_));
                const int x = static_cast<int>(
                    cursor % static_cast<std::size_t>(width_));
                const int rowCells = std::min(
                    width_ - x,
                    static_cast<int>(pageEnd - cursor));
                if (y >= firstCellY && y < lastCellY &&
                    x < lastCellX &&
                    x + rowCells > firstCellX) {
                    intersectsKeptArea = true;
                    break;
                }
                cursor += static_cast<std::size_t>(rowCells);
            }
            if (intersectsKeptArea) {
                continue;
            }
            if (std::all_of(
                    page->begin(), page->end(),
                    [&](const T& value) {
                        return value == defaultValue_;
                    })) {
                page.reset();
            }
        }
    }

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator() = default;
        const_iterator(const SparseGrid* grid, std::size_t index)
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
        const SparseGrid* grid_ = nullptr;
        std::size_t index_ = 0;
    };

    [[nodiscard]] const_iterator begin() const {
        return const_iterator(this, 0);
    }
    [[nodiscard]] const_iterator end() const {
        return const_iterator(this, size());
    }

    friend bool operator==(const SparseGrid& first,
                           const SparseGrid& second) {
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
    [[nodiscard]] std::pair<std::size_t, std::size_t>
    pageAddress(std::size_t index) const {
        return {
            index / static_cast<std::size_t>(pageCellCount),
            index % static_cast<std::size_t>(pageCellCount),
        };
    }

    int width_ = 0;
    int height_ = 0;
    T defaultValue_{};
    std::vector<std::unique_ptr<Page>> pages_;
};

enum class Material : std::uint8_t {
    air,
    dirt,
    sand,
    rock,
    water,
    oil,
    fire,
    smoke,
    steam,
    wood,
    stone,
    metal,
};

struct InputState {
    bool moveLeft = false;
    bool moveRight = false;
    bool jump = false;
    bool swimUp = false;
    bool grappleToggle = false;
    bool reelIn = false;
    bool reelOut = false;
    bool fire = false;
    bool throwGrenade = false;
    bool paint = false;
    bool liquidDebug = false;
    Material paintMaterial = Material::water;
    Vec2 aim{};
};

struct Player {
    Vec2 position{};
    Vec2 velocity{};
    float facing = 1.0F;
    bool onGround = false;
    float health = 100.0F;
    float wetness = 0.0F;
    float oiliness = 0.0F;
    float burning = 0.0F;
    float suffocation = 0.0F;
    float waterSubmersion = 0.0F;
    float oilSubmersion = 0.0F;
};

struct Projectile {
    Vec2 position{};
    Vec2 velocity{};
    float lifetime = 0.0F;
    float energy = 1.0F;
    float submersion = 0.0F;
};

struct Particle {
    Vec2 position{};
    Vec2 velocity{};
    std::array<float, 3> color{};
    float lifetime = 0.0F;
    float maximumLifetime = 0.0F;
    float size = 1.0F;
};

struct Grapple {
    bool active = false;
    bool attached = false;
    Vec2 hookPosition{};
    Vec2 hookVelocity{};
    Vec2 anchor{};
    float ropeLength = 0.0F;
    float wrapCooldown = 0.0F;
    int anchorCellX = 0;
    int anchorCellY = 0;
    // Ordered from the player toward the fixed hook anchor.
    std::vector<Vec2> wrapPoints;
    std::vector<Vec2> points;
    std::vector<Vec2> previousPoints;
};

class World {
public:
#ifdef GUNPOWDER_TEST_SCALE
    static constexpr int simulationScale = 1;
    static constexpr int width = 1024;
    static constexpr int height = 576;
#else
    // 960x540 material pixels at the default 1920x1080 display. This keeps
    // the Noita-style discrete cells visually small while the active-chunk
    // work continues toward affordable 1:1 simulation.
    static constexpr int simulationScale = 3;
    // The camera sees 960x540 simulation cells. This provides roughly
    // 34 screens of horizontal travel and 30 screens from the upper sky to
    // the bedrock, while lazy pages keep untouched material state unallocated.
    static constexpr int width = 32768;
    static constexpr int height = 16384;
#endif
    static constexpr int viewWidth = 320 * simulationScale;
    static constexpr int viewHeight = 180 * simulationScale;
    static constexpr int chunkSize = 64;

    World();

    void regenerate();
    void update(float dt, const InputState& input);

    [[nodiscard]] Material cell(int x, int y) const;
    [[nodiscard]] const Player& player() const { return player_; }
    [[nodiscard]] const std::vector<Projectile>& bullets() const { return bullets_; }
    [[nodiscard]] const std::vector<Projectile>& grenades() const { return grenades_; }
    [[nodiscard]] const std::vector<Particle>& particles() const { return particles_; }
    void setGpuParticlesEnabled(bool enabled);
    [[nodiscard]] std::vector<Particle> takeParticleSpawns();
    void setGpuMaterialSimulationEnabled(bool enabled);
    [[nodiscard]] const Grapple& grapple() const { return grapple_; }
    [[nodiscard]] const SparseGrid<Material>& materials() const {
        return cells_;
    }
    [[nodiscard]] const SparseGrid<std::uint16_t>& liquidAmounts() const {
        return liquidAmount_;
    }
    [[nodiscard]] const SparseGrid<std::int8_t>& liquidFlowX() const {
        return liquidFlowX_;
    }
    [[nodiscard]] const SparseGrid<std::int8_t>& liquidFlowY() const {
        return liquidFlowY_;
    }
    [[nodiscard]] const SparseGrid<std::uint8_t>& liquidFoam() const {
        return liquidFoam_;
    }
    [[nodiscard]] const SparseGrid<float>& heat() const { return heat_; }
    [[nodiscard]] const SparseGrid<float>& burnProgress() const {
        return burnProgress_;
    }
    [[nodiscard]] Material selectedMaterial() const { return selectedMaterial_; }
    [[nodiscard]] bool liquidDebug() const { return liquidDebug_; }
    [[nodiscard]] const MaterialSimulationTimings&
    materialSimulationTimings() const {
        return materialSimulationTimings_;
    }
    [[nodiscard]] Vec2 cameraTopLeft() const;
    [[nodiscard]] Vec2 renderCameraTopLeft() const;
    [[nodiscard]] float explosionFlash() const { return explosionFlash_; }
    [[nodiscard]] int skyOccluderY(int x) const {
        return x >= 0 && x < width
                   ? skyOccluderY_[static_cast<std::size_t>(x)]
                   : 0;
    }
    [[nodiscard]] bool hasInteriorBackdrop(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height &&
               interiorBackdrop_[static_cast<std::size_t>(
                   y * width + x)] != 0;
    }
    [[nodiscard]] std::uint64_t solidRevision() const {
        return solidRevision_;
    }
    void buildDirectionalSunHorizon(
        Vec2 direction, float samplesPerCell,
        std::vector<float>& depths,
        std::vector<std::int32_t>& blockers,
        float& minimumPerpendicularCoordinate,
        Vec2 receiverMinimum = {},
        Vec2 receiverMaximum = {
            static_cast<float>(width),
            static_cast<float>(height),
        }) const;
#ifdef GUNPOWDER_TEST_SCALE
    void setCellForTest(int x, int y, Material material) {
        setCell(x, y, material);
    }
    void setPlayerForTest(Vec2 position, Vec2 velocity = {}) {
        player_.position = position;
        player_.velocity = velocity;
        player_.onGround = false;
        releaseGrapple();
    }
#endif

private:
    friend class VulkanRenderer;

    struct ActiveBounds {
        int minX;
        int maxX;
        int minY;
        int maxY;
    };

    struct Exposure {
        float water = 0.0F;
        float oil = 0.0F;
        float fire = 0.0F;
        float smoke = 0.0F;
        float heat = 0.0F;
    };

    struct LiquidEqualizationMove {
        std::size_t source = 0;
        std::size_t destination = 0;
        std::size_t support = 0;
        Material material = Material::air;
        std::uint8_t phase = 0;
    };

    [[nodiscard]] bool isSolid(int x, int y) const;
    [[nodiscard]] bool overlapsTerrain(Vec2 center, Vec2 halfSize) const;
    void ensureTerrainGenerated(const ActiveBounds& bounds);
    void ensureTerrainChunk(int chunkX, int chunkY);
    [[nodiscard]] int proceduralSurfaceY(int x) const;
    [[nodiscard]] Material proceduralMaterial(int x, int y) const;
    void setCell(int x, int y, Material material);
    void swapCells(int firstX, int firstY, int secondX, int secondY);
    void markMaterialActive(int x, int y);
    [[nodiscard]] bool materialChunkActive(int x, int y) const;
    void ageMaterialChunks();
    void rebuildDirtySkyColumns();
    void captureInteriorBackdrop();
    void destroyCircle(Vec2 center, float radius);
    void explode(Vec2 center, float radius);
    void paintCircle(Vec2 center, float radius, Material material);
    void updatePlayer(float dt, const InputState& input);
    void movePlayerWithCollisions(Vec2 displacement);
    [[nodiscard]] bool tryPlayerStepUp(float horizontalMovement,
                                       float maximumHeight);
    [[nodiscard]] bool snapPlayerToGround(float maximumDistance);
    void updateGrapple(float dt, const InputState& input);
    void updateGrappleWraps(float dt);
    [[nodiscard]] bool ropeLineClear(Vec2 from, Vec2 to) const;
    [[nodiscard]] bool findRopeWrapPoint(Vec2 from, Vec2 to,
                                         Vec2& wrapPoint) const;
    void initializeGrappleChain();
    void simulateGrappleChain(float dt);
    void releaseGrapple();
    [[nodiscard]] Exposure sampleExposure(Vec2 center, Vec2 halfSize) const;
    void updatePlayerEnvironment(float dt, const InputState& input,
                                 const Exposure& exposure);
    void updateBullets(float dt);
    void updateGrenades(float dt);
    void updateMaterials();
    void releaseEmptySimulationPages();
    void cacheLiquidColumnHeads(const ActiveBounds& bounds);
    void updateLiquids(const ActiveBounds& bounds,
                       bool waterOnly = false);
    void prepareLiquidEqualization(const ActiveBounds& bounds);
    void applyLiquidEqualizationPhase(const ActiveBounds& bounds,
                                      int phase);
    void updateHeat();
    void updateCamera(float dt);
    void updateParticles(float dt);
    void emitParticle(Particle particle);
    void spawnExplosionParticles(Vec2 center, float radius);
    void spawnSplashParticles(Vec2 center, Material material, float intensity,
                              Vec2 inheritedVelocity);
    void displaceLiquid(Vec2 center, Vec2 halfSize, Vec2 motion,
                        float strength);
    [[nodiscard]] ActiveBounds activeBounds() const;
    void fireBullet(Vec2 direction);
    void throwGrenade(Vec2 direction);

    SparseGrid<Material> cells_;
    SparseGrid<std::uint16_t> liquidAmount_;
    SparseGrid<std::int8_t> liquidFlowX_;
    SparseGrid<std::int8_t> liquidFlowY_;
    // Consecutive same-liquid cells above this cell, capped to one byte.
    // Rebuilt only inside the active simulation bounds once per material tick.
    SparseGrid<std::uint8_t> liquidHeadDepth_;
    SparseGrid<std::uint8_t> liquidFoam_;
    SparseGrid<std::uint8_t> liquidEqualizationReservation_;
    SparseGrid<float> heat_;
    SparseGrid<float> nextHeat_;
    SparseGrid<float> burnProgress_;
    SparseGrid<float> gasLifetime_;
    SparseGrid<std::int8_t> gasDrift_;
    // Granular cells retain continuous free-fall state even though their
    // authoritative positions remain on the material grid.
    SparseGrid<float> granularVelocityY_;
    SparseGrid<float> granularFallRemainder_;
    SparseGrid<std::uint8_t> moved_;
    SparseGrid<std::uint32_t> liquidFrontierStamp_;
    SparseGrid<std::uint32_t> liquidComponentStamp_;
    std::vector<std::size_t> liquidWorklist_;
    std::vector<std::size_t> liquidNextWorklist_;
    std::uint32_t liquidFrontierGeneration_ = 0;
    std::uint32_t liquidComponentGeneration_ = 0;
    bool rebuildLiquidWorklist_ = true;
    std::vector<std::size_t> liquidComponentQueue_;
    std::vector<std::size_t> liquidHighSurfaces_;
    std::vector<std::size_t> liquidLowSurfaces_;
    std::vector<std::size_t> liquidReservedCells_;
    std::vector<LiquidEqualizationMove> liquidEqualizationMoves_;
    MaterialSimulationTimings materialSimulationTimings_{};
    float currentLiquidSelectionMs_ = 0.0F;
    float currentLiquidGravityMs_ = 0.0F;
    float currentLiquidLateralMs_ = 0.0F;
    float currentLiquidFrontierMs_ = 0.0F;
    float currentLiquidEqualizationMs_ = 0.0F;
    std::uint32_t currentLiquidCandidateVisits_ = 0;
    std::uint32_t currentEqualizedComponents_ = 0;
    std::uint32_t currentEqualizedCells_ = 0;
    std::vector<Material> columnHeadMaterial_;
    std::vector<std::uint8_t> columnHeadDepth_;
    std::vector<std::uint8_t> materialChunkActivity_;
    std::vector<int> skyOccluderY_;
    std::vector<std::uint8_t> skyColumnDirty_;
    SparseGrid<std::uint8_t> interiorBackdrop_;
    std::vector<std::uint8_t> generatedTerrainChunks_;
    int generationOffsetX_ = 0;
    int generationOffsetY_ = 0;
    bool useGenerationCoordinates_ = false;
    Player player_{};
    std::vector<Projectile> bullets_;
    std::vector<Projectile> grenades_;
    std::vector<Particle> particles_;
    std::vector<Particle> particleSpawns_;
    bool gpuParticlesEnabled_ = false;
    Grapple grapple_{};
    std::mt19937 random_{0xC0FFEEU};
    float fireCooldown_ = 0.0F;
    float grenadeCooldown_ = 0.0F;
    float materialAccumulator_ = 0.0F;
    float sparseCleanupAccumulator_ = 0.0F;
    std::uint32_t pendingGpuMaterialSteps_ = 0;
    bool gpuMaterialSimulationEnabled_ = false;
    Material selectedMaterial_ = Material::water;
    bool liquidDebug_ = false;
    Vec2 cameraCenter_{};
    Vec2 cameraShake_{};
    float shakeStrength_ = 0.0F;
    float explosionFlash_ = 0.0F;
    float playerSplashAccumulator_ = 0.0F;
    float playerLiquidDisplacementAccumulator_ = 0.0F;
    float statusParticleAccumulator_ = 0.0F;
    std::uint64_t solidRevision_ = 0;
};

} // namespace gunpowder
