#pragma once

#include "game/chunk_grid.hpp"
#include "game/math.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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
    std::uint32_t activeGranularChunks = 0;
    std::uint32_t activeLiquidChunks = 0;
    std::uint32_t activeGasChunks = 0;
    std::uint32_t activeThermalChunks = 0;
    std::uint32_t activeGranularMicrotiles = 0;
    std::uint32_t activeLiquidMicrotiles = 0;
    std::uint32_t activeGasMicrotiles = 0;
    std::uint32_t activeThermalMicrotiles = 0;
    std::uint32_t materialWorkerThreads = 1;
    std::uint32_t parallelGranularChunks = 0;
    std::uint32_t parallelLiquidChunks = 0;
    std::uint32_t parallelGasChunks = 0;
    std::uint32_t parallelThermalChunks = 0;
    std::uint32_t granularMoveProposals = 0;
    std::uint32_t granularMovesAccepted = 0;
    std::uint32_t granularMoveConflicts = 0;
    std::uint32_t gasMoveProposals = 0;
    std::uint32_t gasMovesAccepted = 0;
    std::uint32_t gasMoveConflicts = 0;
    std::uint32_t liquidMoveProposals = 0;
    std::uint32_t liquidMovesAccepted = 0;
    std::uint32_t liquidMoveConflicts = 0;
    std::uint32_t liquidGravityConflicts = 0;
    std::uint32_t liquidLateralConflicts = 0;
    std::uint32_t parallelLiquidColumnVisits = 0;
    std::uint32_t parallelLiquidVerticalMoves = 0;
    std::uint32_t liquidPreparationCellVisits = 0;
    std::uint32_t liquidHeadSummaryHits = 0;
    std::uint32_t liquidEqualizationSeedVisits = 0;
    std::uint32_t liquidCandidateVisits = 0;
    std::uint32_t equalizedComponents = 0;
    std::uint32_t equalizedCells = 0;
    bool valid = false;
};

struct SolidDirtyRegion {
    int minX = 0;
    int minY = 0;
    int maxX = -1;
    int maxY = -1;

    [[nodiscard]] bool valid() const {
        return minX <= maxX && minY <= maxY;
    }
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
    [[nodiscard]] SolidDirtyRegion consumeSolidDirtyRegion();
    void buildDirectionalSunHorizon(
        Vec2 direction, float samplesPerCell,
        std::vector<float>& depths,
        std::vector<std::int32_t>& blockers,
        float& minimumPerpendicularCoordinate,
        Vec2 receiverMinimum = {},
        Vec2 receiverMaximum = {
            static_cast<float>(width),
            static_cast<float>(height),
        },
        const SolidDirtyRegion* dirtyRegion = nullptr,
        bool parallelBuild = false) const;
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
    void clearMaterialActivityForTest();
    [[nodiscard]] std::array<bool, 4>
    materialActivityForTest(int x, int y) const;
    [[nodiscard]] std::array<bool, 4>
    materialMicrotileActivityForTest(int x, int y) const;
    [[nodiscard]] std::uint8_t
    liquidHeadDepthForTest(int x, int y) const {
        return x >= 0 && x < width &&
                       y >= 0 && y < height
                   ? liquidHeadDepth_[static_cast<std::size_t>(
                         y * width + x)]
                   : 0;
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

    static constexpr std::uint8_t granularActivity = 1U << 0U;
    static constexpr std::uint8_t liquidActivity = 1U << 1U;
    static constexpr std::uint8_t gasActivity = 1U << 2U;
    static constexpr std::uint8_t thermalActivity = 1U << 3U;
    static constexpr std::uint8_t allMaterialActivity =
        granularActivity | liquidActivity |
        gasActivity | thermalActivity;
    static constexpr std::size_t materialActivitySystemCount = 4;
    static constexpr int materialMicrotileSize = 8;
    static constexpr int materialMicrotilesPerAxis =
        chunkSize / materialMicrotileSize;
    static constexpr int materialMicrotilesPerChunk =
        materialMicrotilesPerAxis * materialMicrotilesPerAxis;
    static_assert(chunkSize % materialMicrotileSize == 0);
    static_assert(materialMicrotilesPerChunk == 64);

    struct MaterialChunkActivity {
        std::array<std::uint8_t,
                   materialActivitySystemCount> lifetime{};
        std::array<std::uint64_t,
                   materialActivitySystemCount> microtiles{};
    };

    struct LiquidChunkColumnSummary {
        std::array<Material, chunkSize> bottomMaterial{};
        std::array<std::uint8_t, chunkSize> bottomDepth{};
        std::uint32_t generation = 0;
    };

    [[nodiscard]] bool isSolid(int x, int y) const;
    [[nodiscard]] bool overlapsTerrain(Vec2 center, Vec2 halfSize) const;
    void ensureTerrainGenerated(const ActiveBounds& bounds);
    void ensureTerrainChunk(int chunkX, int chunkY);
    [[nodiscard]] int proceduralSurfaceY(int x) const;
    [[nodiscard]] Material proceduralMaterial(int x, int y) const;
    void setCell(int x, int y, Material material);
    void swapCells(int firstX, int firstY, int secondX, int secondY);
    [[nodiscard]] static std::uint8_t
    materialActivityMask(Material material);
    void markMaterialActive(
        int x, int y,
        std::uint8_t activityMask = allMaterialActivity);
    void invalidateSettledLiquidNear(int x, int y);
    [[nodiscard]] bool
    liquidCellBelongsToSettledComponent(std::size_t index) const;
    [[nodiscard]] bool materialChunkActive(
        int x, int y,
        std::uint8_t activityMask = allMaterialActivity) const;
    [[nodiscard]] bool materialMicrotileActive(
        int x, int y,
        std::uint8_t activityMask = allMaterialActivity) const;
    void ageMaterialChunks();
    void markSolidDirty(int minX, int minY, int maxX, int maxY);
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
    SparseGrid<std::uint32_t> liquidSettledComponent_;
    std::vector<std::size_t> liquidWorklist_;
    std::vector<std::size_t> liquidNextWorklist_;
    std::vector<std::size_t> thermalWorklist_;
    std::uint32_t liquidFrontierGeneration_ = 0;
    std::uint32_t liquidComponentGeneration_ = 0;
    std::uint32_t nextSettledLiquidComponent_ = 1;
    std::vector<std::uint8_t> settledLiquidComponentValid_{0};
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
    std::uint32_t currentLiquidPreparationCellVisits_ = 0;
    std::uint32_t currentLiquidHeadSummaryHits_ = 0;
    std::uint32_t currentLiquidEqualizationSeedVisits_ = 0;
    std::uint32_t currentEqualizedComponents_ = 0;
    std::uint32_t currentEqualizedCells_ = 0;
    std::uint32_t currentLiquidMoveProposals_ = 0;
    std::uint32_t currentLiquidMovesAccepted_ = 0;
    std::uint32_t currentLiquidMoveConflicts_ = 0;
    std::uint32_t currentLiquidGravityConflicts_ = 0;
    std::uint32_t currentLiquidLateralConflicts_ = 0;
    std::uint32_t currentParallelLiquidColumnVisits_ = 0;
    std::uint32_t currentParallelLiquidVerticalMoves_ = 0;
    std::vector<MaterialChunkActivity> materialChunkActivity_;
    std::vector<std::unique_ptr<LiquidChunkColumnSummary>>
        liquidChunkColumnSummaries_;
    std::uint32_t liquidPreparationGeneration_ = 0;
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
    bool materialScanRight_ = true;
    std::uint64_t materialStep_ = 0;
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
    SolidDirtyRegion solidDirtyRegion_{};
};

} // namespace gunpowder
