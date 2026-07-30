#include "game/world.hpp"
#include "game/parallel_executor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gunpowder {
namespace {

constexpr float spatialScale = static_cast<float>(World::simulationScale);
constexpr float scaled(float value) { return value * spatialScale; }
constexpr int scaledCell(int value) {
    return value * World::simulationScale;
}

constexpr Vec2 playerHalfSize{scaled(2.2F), scaled(4.2F)};
constexpr float gravity = scaled(95.0F);
constexpr float terminalFallSpeed = scaled(90.0F);
constexpr float materialTimeStep = 1.0F / 30.0F;
constexpr float maximumRunSpeed = scaled(46.0F);
constexpr float playerStepHeight = scaled(1.35F);
constexpr float playerGroundSnapDistance = scaled(1.45F);
constexpr float playerGroundProbeIncrement = scaled(0.125F);
constexpr std::uint16_t maximumLiquidMass = 255;

float elapsedMilliseconds(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - begin).count();
}

void smoothTiming(float& destination, float sample) {
    destination =
        destination <= 0.0F
            ? sample
            : destination * 0.85F + sample * 0.15F;
}

bool isLiquid(Material material) {
    return material == Material::water || material == Material::oil;
}

bool isSkyOccluder(Material material) {
    return material == Material::dirt ||
           material == Material::sand ||
           material == Material::rock ||
           material == Material::wood ||
           material == Material::stone ||
           material == Material::metal;
}

std::size_t indexOf(int x, int y) {
    return static_cast<std::size_t>(y * World::width + x);
}

ParallelExecutor& materialExecutor() {
    static ParallelExecutor executor;
    return executor;
}

} // namespace

World::World()
    : cells_(width, height, Material::air),
      liquidAmount_(width, height, 0),
      liquidFlowX_(width, height, 0),
      liquidFlowY_(width, height, 0),
      liquidHeadDepth_(width, height, 0),
      liquidFoam_(width, height, 0),
      liquidEqualizationReservation_(width, height, 0),
      heat_(width, height, 0.0F),
      nextHeat_(width, height, 0.0F),
      burnProgress_(width, height, 0.0F),
      gasLifetime_(width, height, 0.0F),
      gasDrift_(width, height, 0),
      granularVelocityY_(width, height, 0.0F),
      granularFallRemainder_(width, height, 0.0F),
      moved_(width, height, 0),
      liquidFrontierStamp_(width, height, 0),
      liquidComponentStamp_(width, height, 0),
      materialChunkActivity_(
          static_cast<std::size_t>(
              ((width + chunkSize - 1) / chunkSize) *
              ((height + chunkSize - 1) / chunkSize))),
      liquidChunkColumnSummaries_(
          static_cast<std::size_t>(
              ((width + chunkSize - 1) / chunkSize) *
              ((height + chunkSize - 1) / chunkSize))),
      skyOccluderY_(static_cast<std::size_t>(width), height),
      skyColumnDirty_(static_cast<std::size_t>(width), 1),
      interiorBackdrop_(width, height, 0),
      generatedTerrainChunks_(
          static_cast<std::size_t>(
              ((width + chunkSize - 1) / chunkSize) *
              ((height + chunkSize - 1) / chunkSize)),
          0) {
    regenerate();
}

void World::regenerate() {
    cells_.reset(Material::air);
    liquidAmount_.reset(0);
    liquidFlowX_.reset(0);
    liquidFlowY_.reset(0);
    liquidHeadDepth_.reset(0);
    liquidFoam_.reset(0);
    liquidEqualizationReservation_.reset(0);
    heat_.reset(0.0F);
    nextHeat_.reset(0.0F);
    burnProgress_.reset(0.0F);
    gasLifetime_.reset(0.0F);
    gasDrift_.reset(0);
    granularVelocityY_.reset(0.0F);
    granularFallRemainder_.reset(0.0F);
    moved_.reset(0);
    liquidFrontierStamp_.reset(0);
    liquidComponentStamp_.reset(0);
    liquidFrontierGeneration_ = 0;
    liquidComponentGeneration_ = 0;
    liquidWorklist_.clear();
    liquidNextWorklist_.clear();
    thermalWorklist_.clear();
    liquidReservedCells_.clear();
    for (auto& summary : liquidChunkColumnSummaries_) {
        summary.reset();
    }
    liquidPreparationGeneration_ = 0;
    std::fill(materialChunkActivity_.begin(),
              materialChunkActivity_.end(),
              MaterialChunkActivity{});
    interiorBackdrop_.reset(0);
    std::fill(generatedTerrainChunks_.begin(),
              generatedTerrainChunks_.end(), 0);
    generationOffsetX_ =
        width > 4096 ? width / 2 - scaledCell(512) : 0;
    generationOffsetY_ =
        height > 4096 ? height / 4 - scaledCell(112) : 0;
    useGenerationCoordinates_ = false;
    for (int x = 0; x < width; ++x) {
        skyOccluderY_[static_cast<std::size_t>(x)] =
            proceduralSurfaceY(x);
    }
    std::fill(skyColumnDirty_.begin(), skyColumnDirty_.end(), 0);
    bullets_.clear();
    grenades_.clear();
    particles_.clear();
    particleSpawns_.clear();
    pendingGpuMaterialSteps_ = 0;
    materialScanRight_ = true;
    grapple_ = Grapple{};
    playerSplashAccumulator_ = 0.0F;
    playerLiquidDisplacementAccumulator_ = 0.0F;

    const int generationWidth =
        std::min(width, scaledCell(1024));
    const int generationHeight =
        std::min(height - generationOffsetY_, scaledCell(576));
    ensureTerrainGenerated({
        generationOffsetX_,
        generationOffsetX_ + generationWidth,
        std::max(0, generationOffsetY_),
        std::min(height, generationOffsetY_ + generationHeight),
    });
    useGenerationCoordinates_ = true;

    std::uniform_real_distribution<float> noise(scaled(-4.0F),
                                                 scaled(4.0F));
    for (int x = 0; x < generationWidth; ++x) {
        const float logicalX =
            static_cast<float>(x) / spatialScale;
        const float wave = std::sin(logicalX * 0.055F) * scaled(9.0F);
        const int surface =
            scaledCell(112) + static_cast<int>(wave + noise(random_));
        for (int y = surface; y < generationHeight; ++y) {
            const bool localBedrock =
                height <= scaledCell(576) &&
                y > generationHeight - scaledCell(8);
            Material material =
                localBedrock ? Material::rock : Material::dirt;
            if (y < surface + scaledCell(7) &&
                ((x * 17 + y * 31) % 11) < 4) {
                material = Material::sand;
            }
            setCell(x, y, material);
        }
    }

    destroyCircle({scaled(92.0F), scaled(139.0F)}, scaled(20.0F));
    destroyCircle({scaled(198.0F), scaled(132.0F)}, scaled(15.0F));
    destroyCircle({scaled(248.0F), scaled(151.0F)}, scaled(12.0F));

    // Carve a connected-looking field of underground caverns. Regenerating
    // advances the seeded generator, producing a repeatable sequence of maps.
    std::uniform_real_distribution<float> caveX(
        scaled(120.0F),
        static_cast<float>(generationWidth) - scaled(80.0F));
    std::uniform_real_distribution<float> caveY(
        scaled(155.0F),
        static_cast<float>(generationHeight) - scaled(45.0F));
    std::uniform_real_distribution<float> caveAngle(-3.14159F, 3.14159F);
    std::uniform_real_distribution<float> caveTurn(-0.42F, 0.42F);
    std::uniform_real_distribution<float> caveRadius(scaled(6.0F),
                                                      scaled(13.0F));
    for (int tunnel = 0; tunnel < 24; ++tunnel) {
        Vec2 position{caveX(random_), caveY(random_)};
        float angle = caveAngle(random_);
        for (int segment = 0; segment < 48; ++segment) {
            destroyCircle(position, caveRadius(random_));
            position.x += std::cos(angle) * scaled(7.0F);
            position.y += std::sin(angle) * scaled(5.0F);
            position.x = std::clamp(
                position.x, scaled(24.0F),
                static_cast<float>(generationWidth) - scaled(24.0F));
            position.y = std::clamp(
                position.y, scaled(135.0F),
                static_cast<float>(generationHeight) - scaled(18.0F));
            angle += caveTurn(random_);
        }
    }

    // Keep the opening area readable and safe before the procedural terrain
    // takes over.
    for (int x = scaledCell(18); x < scaledCell(118); ++x) {
        for (int y = scaledCell(96); y < scaledCell(108); ++y) {
            setCell(x, y, Material::air);
        }
        for (int y = scaledCell(108); y < scaledCell(112); ++y) {
            setCell(x, y, Material::dirt);
        }
    }

    // Seed two test pools so the material interactions are visible immediately.
    for (int y = scaledCell(143); y < scaledCell(157); ++y) {
        for (int x = scaledCell(72); x < scaledCell(112); ++x) {
            const float dx =
                static_cast<float>(x) + 0.5F - scaled(92.0F);
            const float dy =
                static_cast<float>(y) + 0.5F - scaled(139.0F);
            if (dx * dx + dy * dy < scaled(19.0F) * scaled(19.0F) &&
                cell(x, y) == Material::air) {
                setCell(x, y, Material::water);
            }
        }
    }
    for (int y = scaledCell(134); y < scaledCell(145); ++y) {
        for (int x = scaledCell(184); x < scaledCell(212); ++x) {
            const float dx =
                static_cast<float>(x) + 0.5F - scaled(198.0F);
            const float dy =
                static_cast<float>(y) + 0.5F - scaled(132.0F);
            if (dx * dx + dy * dy < scaled(14.0F) * scaled(14.0F) &&
                cell(x, y) == Material::air) {
                setCell(x, y, Material::oil);
            }
        }
    }

    for (int x = scaledCell(145); x < scaledCell(180); ++x) {
        for (int y = scaledCell(94); y < scaledCell(98); ++y) {
            setCell(x, y, Material::rock);
        }
    }

    const auto fillRectangle = [&](int left, int top, int right, int bottom,
                                   Material material) {
        for (int y = scaledCell(top); y < scaledCell(bottom); ++y) {
            for (int x = scaledCell(left); x < scaledCell(right); ++x) {
                setCell(x, y, material);
            }
        }
    };

    // The opening biome is a weathered tomb rather than a natural cave. Its
    // descending rooms preserve the starter ledge used by movement tutorials,
    // then open into a pillared burial hall and a flooded cistern.
    fillRectangle(18, 54, 118, 108, Material::air);
    fillRectangle(118, 46, 228, 140, Material::air);
    fillRectangle(228, 58, 320, 154, Material::air);

    // Entry chamber.
    fillRectangle(12, 48, 118, 55, Material::stone);
    fillRectangle(12, 48, 19, 113, Material::stone);
    fillRectangle(18, 108, 118, 114, Material::stone);
    fillRectangle(69, 101, 92, 108, Material::wood); // sarcophagus
    fillRectangle(66, 99, 95, 102, Material::stone);
    fillRectangle(72, 99, 89, 101, Material::metal);
    fillRectangle(73, 102, 75, 108, Material::metal);
    fillRectangle(86, 102, 88, 108, Material::metal);

    // Lower burial hall, with stone columns and wooden staging left behind by
    // grave robbers. The first room drops into this hall at x=118.
    fillRectangle(118, 40, 228, 47, Material::stone);
    fillRectangle(118, 140, 228, 146, Material::stone);
    fillRectangle(148, 115, 158, 140, Material::stone);
    fillRectangle(145, 111, 161, 116, Material::stone);
    fillRectangle(198, 108, 207, 140, Material::stone);
    fillRectangle(195, 104, 210, 109, Material::stone);
    fillRectangle(124, 119, 146, 122, Material::wood);
    fillRectangle(128, 122, 131, 140, Material::wood);
    fillRectangle(163, 101, 193, 104, Material::wood);
    fillRectangle(187, 104, 190, 140, Material::wood);

    // An intact oil basin provides fuel the player can deliberately ignite.
    fillRectangle(174, 130, 179, 140, Material::stone);
    fillRectangle(194, 130, 199, 140, Material::stone);
    fillRectangle(174, 136, 199, 140, Material::stone);
    fillRectangle(179, 130, 194, 136, Material::oil);

    // Cistern chamber. A burnable wooden gate separates it from the hall, and
    // a bridge crosses water stored behind a low stone retaining wall.
    fillRectangle(228, 52, 320, 59, Material::stone);
    fillRectangle(314, 52, 321, 160, Material::stone);
    fillRectangle(228, 154, 321, 160, Material::stone);
    fillRectangle(222, 52, 229, 87, Material::stone);
    fillRectangle(222, 127, 229, 146, Material::stone);
    fillRectangle(223, 87, 228, 127, Material::wood);
    fillRectangle(247, 132, 253, 154, Material::stone);
    fillRectangle(253, 140, 314, 154, Material::water);
    fillRectangle(230, 123, 306, 126, Material::wood);
    fillRectangle(238, 126, 241, 154, Material::wood);
    fillRectangle(298, 126, 301, 154, Material::wood);

    // Continue the hand-authored opening with a seeded chain of tomb modules.
    // The entry rooms remain stable teaching spaces, while everything beyond
    // the cistern varies on regeneration. Adjacent modules overlap their
    // structural walls, then receive an explicitly carved doorway so every
    // generated wing has a traversable main route.
    struct TombModule {
        int left = 0;
        int top = 0;
        int right = 0;
        int bottom = 0;
        int variant = 0;
    };
    struct TombDoor {
        int wallX = 0;
        int centerY = 0;
    };
    std::vector<TombModule> tombModules;
    std::vector<TombDoor> tombDoors;
    tombModules.reserve(10);
    tombDoors.reserve(10);
    TombModule previousModule{
        .left = 228,
        .top = 52,
        .right = 321,
        .bottom = 160,
        .variant = -1,
    };
    std::uniform_int_distribution<int> roomWidth(72, 108);
    std::uniform_int_distribution<int> roomHeight(62, 92);
    std::uniform_int_distribution<int> floorChange(-18, 18);
    std::uniform_int_distribution<int> roomVariant(0, 4);
    int nextLeft = previousModule.right - 1;
    for (int roomIndex = 0;
         roomIndex < 9 && nextLeft < 950; ++roomIndex) {
        const int generatedWidth = std::min(
            roomWidth(random_), 980 - nextLeft);
        if (generatedWidth < 48) {
            break;
        }
        const int generatedBottom = std::clamp(
            previousModule.bottom + floorChange(random_),
            142, 255);
        const int generatedHeight = roomHeight(random_);
        const TombModule module{
            .left = nextLeft,
            .top = generatedBottom - generatedHeight,
            .right = nextLeft + generatedWidth,
            .bottom = generatedBottom,
            .variant = roomVariant(random_),
        };

        fillRectangle(module.left, module.top,
                      module.right, module.bottom,
                      Material::stone);
        fillRectangle(module.left + 4, module.top + 4,
                      module.right - 4, module.bottom - 4,
                      Material::air);

        const int overlapTop =
            std::max(previousModule.top, module.top) + 13;
        const int overlapBottom =
            std::min(previousModule.bottom, module.bottom) - 13;
        tombDoors.push_back({
            .wallX = module.left,
            .centerY = overlapTop <= overlapBottom
                           ? (overlapTop + overlapBottom) / 2
                           : (module.top + module.bottom) / 2,
        });

        const int interiorLeft = module.left + 7;
        const int interiorRight = module.right - 7;
        const int floorY = module.bottom - 4;
        const int interiorWidth = interiorRight - interiorLeft;
        switch (module.variant) {
        case 0: {
            // Burial chamber: low sarcophagi leave the upper route clear for
            // grappling while providing destructible wooden cover.
            const int firstLeft = interiorLeft + 8;
            const int secondLeft =
                std::max(firstLeft + 22, interiorRight - 29);
            fillRectangle(firstLeft, floorY - 8,
                          firstLeft + 18, floorY,
                          Material::wood);
            fillRectangle(firstLeft - 2, floorY - 10,
                          firstLeft + 20, floorY - 7,
                          Material::stone);
            if (secondLeft + 20 < interiorRight) {
                fillRectangle(secondLeft, floorY - 8,
                              secondLeft + 18, floorY,
                              Material::wood);
                fillRectangle(secondLeft - 2, floorY - 10,
                              secondLeft + 20, floorY - 7,
                              Material::stone);
            }
            break;
        }
        case 1: {
            // Flooded room with a dry bridge and retaining edges.
            fillRectangle(interiorLeft + 3, floorY - 18,
                          interiorLeft + 8, floorY,
                          Material::stone);
            fillRectangle(interiorRight - 8, floorY - 18,
                          interiorRight - 3, floorY,
                          Material::stone);
            fillRectangle(interiorLeft + 8, floorY - 13,
                          interiorRight - 8, floorY,
                          Material::water);
            fillRectangle(interiorLeft + 4, floorY - 17,
                          interiorRight - 4, floorY - 14,
                          Material::wood);
            if (interiorWidth > 64) {
                const int support = (interiorLeft + interiorRight) / 2;
                fillRectangle(support - 1, floorY - 14,
                              support + 2, floorY,
                              Material::wood);
            }
            break;
        }
        case 2: {
            // Oil vault: a contained fuel source beside wooden staging.
            const int basinLeft =
                interiorLeft + std::max(9, interiorWidth / 3);
            const int basinRight =
                std::min(interiorRight - 7, basinLeft + 27);
            fillRectangle(basinLeft - 4, floorY - 15,
                          basinLeft, floorY, Material::stone);
            fillRectangle(basinRight, floorY - 15,
                          basinRight + 4, floorY, Material::stone);
            fillRectangle(basinLeft - 4, floorY - 4,
                          basinRight + 4, floorY,
                          Material::stone);
            fillRectangle(basinLeft, floorY - 12,
                          basinRight, floorY - 4,
                          Material::oil);
            fillRectangle(interiorLeft + 3, floorY - 24,
                          basinLeft - 5, floorY - 21,
                          Material::wood);
            break;
        }
        case 3: {
            // Collapsed gallery. Loose sand forms an irregular ramp without
            // sealing the generated doorway or the upper grapple route.
            const bool collapseOnLeft =
                (roomIndex & 1) == 0;
            const int pileStart =
                collapseOnLeft
                    ? interiorLeft + 7
                    : interiorRight - 30;
            for (int column = 0; column < 24; ++column) {
                const int height =
                    collapseOnLeft
                        ? std::max(2, 18 - column / 2)
                        : std::max(2, 7 + column / 2);
                fillRectangle(pileStart + column,
                              floorY - height,
                              pileStart + column + 1,
                              floorY, Material::sand);
            }
            break;
        }
        default: {
            // Grave-robber staging at two elevations.
            const int middle =
                (interiorLeft + interiorRight) / 2;
            fillRectangle(interiorLeft + 6, floorY - 20,
                          middle - 3, floorY - 17,
                          Material::wood);
            fillRectangle(interiorLeft + 9, floorY - 17,
                          interiorLeft + 12, floorY,
                          Material::wood);
            fillRectangle(middle + 5, floorY - 35,
                          interiorRight - 6, floorY - 32,
                          Material::wood);
            fillRectangle(interiorRight - 12, floorY - 32,
                          interiorRight - 9, floorY,
                          Material::wood);
            break;
        }
        }

        tombModules.push_back(module);
        previousModule = module;
        nextLeft = module.right - 1;
    }

    // Carve connections after all shells are placed; otherwise the following
    // module would rebuild its overlapping wall over the previous doorway.
    for (const TombDoor& door : tombDoors) {
        fillRectangle(door.wallX - 5, door.centerY - 9,
                      door.wallX + 6, door.centerY + 10,
                      Material::air);
        // A cracked wooden lintel makes the transition readable without
        // blocking the guaranteed route.
        fillRectangle(door.wallX - 4, door.centerY - 12,
                      door.wallX + 5, door.centerY - 9,
                      Material::wood);
    }

    useGenerationCoordinates_ = false;
    player_ = Player{
        .position = {
            scaled(48.0F) +
                static_cast<float>(generationOffsetX_),
            scaled(80.0F) + static_cast<float>(generationOffsetY_),
        },
        .velocity = {},
        .facing = 1.0F,
        .onGround = false,
    };
    cameraCenter_ = player_.position;
    cameraShake_ = {};
    shakeStrength_ = 0.0F;
    explosionFlash_ = 0.0F;
    fireCooldown_ = 0.0F;
    grenadeCooldown_ = 0.0F;
    materialAccumulator_ = 0.0F;
    sparseCleanupAccumulator_ = 0.0F;
    rebuildDirtySkyColumns();
    captureInteriorBackdrop();
}

void World::update(float dt, const InputState& input) {
    ActiveBounds generatedBounds = activeBounds();
    generatedBounds.minX =
        std::max(0, generatedBounds.minX - chunkSize);
    generatedBounds.maxX =
        std::min(width, generatedBounds.maxX + chunkSize);
    generatedBounds.minY =
        std::max(0, generatedBounds.minY - chunkSize);
    generatedBounds.maxY =
        std::min(height, generatedBounds.maxY + chunkSize);
    ensureTerrainGenerated(generatedBounds);

    fireCooldown_ = std::max(0.0F, fireCooldown_ - dt);
    grenadeCooldown_ = std::max(0.0F, grenadeCooldown_ - dt);
    selectedMaterial_ = input.paintMaterial;
    liquidDebug_ = input.liquidDebug;

    const Vec2 aimDirection = normalized(input.aim - player_.position);
    if (input.fire && fireCooldown_ <= 0.0F) {
        fireBullet(aimDirection);
        fireCooldown_ = 0.11F;
    }
    if (input.throwGrenade && grenadeCooldown_ <= 0.0F) {
        throwGrenade(aimDirection);
        grenadeCooldown_ = 0.55F;
    }
    if (input.paint) {
        paintCircle(input.aim, scaled(3.2F), input.paintMaterial);
    }

    updatePlayer(dt, input);
    updateGrapple(dt, input);
    updateBullets(dt);
    updateGrenades(dt);

    materialAccumulator_ += dt;
    while (materialAccumulator_ >= materialTimeStep) {
        // The local full-cell material simulation remains authoritative.
        // Legacy GPU pressure jobs stay disabled in the Noita-style mode.
        const auto simulationBegin =
            std::chrono::steady_clock::now();
        updateMaterials();
        const auto heatBegin =
            std::chrono::steady_clock::now();
        updateHeat();
        const auto simulationEnd =
            std::chrono::steady_clock::now();
        smoothTiming(
            materialSimulationTimings_.heatMs,
            elapsedMilliseconds(heatBegin, simulationEnd));
        smoothTiming(
            materialSimulationTimings_.totalMs,
            elapsedMilliseconds(simulationBegin, simulationEnd));
        materialSimulationTimings_.valid = true;
        if (gpuMaterialSimulationEnabled_) {
            ++pendingGpuMaterialSteps_;
        }
        materialAccumulator_ -= materialTimeStep;
    }
    updateParticles(dt);
    updateCamera(dt);
    rebuildDirtySkyColumns();
    sparseCleanupAccumulator_ += dt;
    if (sparseCleanupAccumulator_ >= 2.0F) {
        sparseCleanupAccumulator_ = 0.0F;
        releaseEmptySimulationPages();
    }
}

Vec2 World::cameraTopLeft() const {
    return {
        std::clamp(cameraCenter_.x - static_cast<float>(viewWidth) * 0.5F,
                   0.0F, static_cast<float>(width - viewWidth)),
        std::clamp(cameraCenter_.y - static_cast<float>(viewHeight) * 0.5F,
                   0.0F, static_cast<float>(height - viewHeight)),
    };
}

Vec2 World::renderCameraTopLeft() const {
    const Vec2 base = cameraTopLeft();
    return {
        std::clamp(base.x + cameraShake_.x, 0.0F,
                   static_cast<float>(width - viewWidth)),
        std::clamp(base.y + cameraShake_.y, 0.0F,
                   static_cast<float>(height - viewHeight)),
    };
}

Material World::cell(int x, int y) const {
    if (useGenerationCoordinates_) {
        x += generationOffsetX_;
        y += generationOffsetY_;
    }
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return Material::rock;
    }
    const_cast<World*>(this)->ensureTerrainChunk(
        x / chunkSize, y / chunkSize);
    return cells_[indexOf(x, y)];
}

bool World::isSolid(int x, int y) const {
    const Material material = cell(x, y);
    return material == Material::dirt || material == Material::sand ||
           material == Material::rock || material == Material::wood ||
           material == Material::stone || material == Material::metal;
}

int World::proceduralSurfaceY(int x) const {
    const float logicalX =
        static_cast<float>(x) / spatialScale;
    const float broad =
        std::sin(logicalX * 0.011F) * scaled(18.0F);
    const float detail =
        std::sin(logicalX * 0.047F + 1.7F) * scaled(6.0F);
    const int baseline =
        height > 4096 ? height / 4 : scaledCell(112);
    return std::clamp(
        baseline + static_cast<int>(broad + detail),
        scaledCell(32), height - scaledCell(64));
}

Material World::proceduralMaterial(int x, int y) const {
    const int surface = proceduralSurfaceY(x);
    if (y < surface) {
        const float logicalX =
            static_cast<float>(x) / spatialScale;
        const float logicalY =
            static_cast<float>(y) / spatialScale;
        const float altitude =
            static_cast<float>(surface - y) / spatialScale;
        const float islandField =
            std::sin(logicalX * 0.031F) +
            std::sin((logicalX + logicalY) * 0.019F) +
            std::sin(logicalY * 0.053F + 2.1F);
        if (altitude > 110.0F && altitude < 920.0F &&
            islandField > 2.62F) {
            return islandField > 2.82F ? Material::rock
                                       : Material::dirt;
        }
        return Material::air;
    }

    if (y >= height - scaledCell(12)) {
        return Material::rock;
    }
    const int depth = y - surface;
    if (depth < scaledCell(7)) {
        return ((x * 17 + y * 31) % 11) < 4
                   ? Material::sand
                   : Material::dirt;
    }

    const float logicalX =
        static_cast<float>(x) / spatialScale;
    const float logicalY =
        static_cast<float>(y) / spatialScale;
    const float caveField =
        std::sin(logicalX * 0.024F) +
        std::sin((logicalX + logicalY) * 0.014F + 0.8F) +
        std::sin(logicalY * 0.038F + 2.6F);
    if (depth > scaledCell(24) && caveField > 2.05F) {
        return Material::air;
    }
    if (depth > scaledCell(420)) {
        return caveField < -2.15F ? Material::rock
                                  : Material::stone;
    }
    if (depth > scaledCell(130)) {
        return caveField < -2.35F ? Material::rock
                                  : Material::dirt;
    }
    return Material::dirt;
}

void World::ensureTerrainGenerated(const ActiveBounds& bounds) {
    const int chunksWide =
        (width + chunkSize - 1) / chunkSize;
    const int chunksHigh =
        (height + chunkSize - 1) / chunkSize;
    const int firstChunkX =
        std::clamp(bounds.minX / chunkSize, 0, chunksWide);
    const int lastChunkX =
        std::clamp((bounds.maxX + chunkSize - 1) / chunkSize,
                   0, chunksWide);
    const int firstChunkY =
        std::clamp(bounds.minY / chunkSize, 0, chunksHigh);
    const int lastChunkY =
        std::clamp((bounds.maxY + chunkSize - 1) / chunkSize,
                   0, chunksHigh);
    for (int chunkY = firstChunkY; chunkY < lastChunkY; ++chunkY) {
        for (int chunkX = firstChunkX; chunkX < lastChunkX; ++chunkX) {
            ensureTerrainChunk(chunkX, chunkY);
        }
    }
}

void World::ensureTerrainChunk(int chunkX, int chunkY) {
    const int chunksWide =
        (width + chunkSize - 1) / chunkSize;
    const int chunksHigh =
        (height + chunkSize - 1) / chunkSize;
    if (chunkX < 0 || chunkX >= chunksWide ||
        chunkY < 0 || chunkY >= chunksHigh) {
        return;
    }
    const std::size_t chunkIndex =
        static_cast<std::size_t>(chunkY * chunksWide + chunkX);
    if (generatedTerrainChunks_[chunkIndex] != 0) {
        return;
    }
    generatedTerrainChunks_[chunkIndex] = 1;

    const int beginX = chunkX * chunkSize;
    const int endX = std::min(width, beginX + chunkSize);
    const int beginY = chunkY * chunkSize;
    const int endY = std::min(height, beginY + chunkSize);
    bool containsSolid = false;
    for (int y = beginY; y < endY; ++y) {
        for (int x = beginX; x < endX; ++x) {
            const Material material = proceduralMaterial(x, y);
            const std::size_t index = indexOf(x, y);
            cells_.set(index, material);
            if (isSkyOccluder(material)) {
                containsSolid = true;
            } else if (y >= proceduralSurfaceY(x)) {
                interiorBackdrop_.set(index, 1);
            }
        }
    }
    if (containsSolid) {
        ++solidRevision_;
    }
}

bool World::overlapsTerrain(Vec2 center, Vec2 halfSize) const {
    const int left = static_cast<int>(std::floor(center.x - halfSize.x));
    const int right = static_cast<int>(std::floor(center.x + halfSize.x));
    const int top = static_cast<int>(std::floor(center.y - halfSize.y));
    const int bottom = static_cast<int>(std::floor(center.y + halfSize.y));
    constexpr float cornerRadius = scaled(0.68F);
    const Vec2 innerHalfSize{
        std::max(0.0F, halfSize.x - cornerRadius),
        std::max(0.0F, halfSize.y - cornerRadius),
    };
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            if (!isSolid(x, y)) {
                continue;
            }
            const float distanceX = std::max(
                std::abs(static_cast<float>(x) + 0.5F - center.x) -
                    (innerHalfSize.x + 0.5F),
                0.0F);
            const float distanceY = std::max(
                std::abs(static_cast<float>(y) + 0.5F - center.y) -
                    (innerHalfSize.y + 0.5F),
                0.0F);
            if (distanceX * distanceX + distanceY * distanceY <=
                cornerRadius * cornerRadius) {
                return true;
            }
        }
    }
    return false;
}

void World::setCell(int x, int y, Material material) {
    if (useGenerationCoordinates_) {
        x += generationOffsetX_;
        y += generationOffsetY_;
    }
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    ensureTerrainChunk(x / chunkSize, y / chunkSize);
    const std::size_t index = indexOf(x, y);
    const Material previous = cells_[index];
    cells_.set(index, material);
    const bool previousOccluder = isSkyOccluder(previous);
    const bool nextOccluder = isSkyOccluder(material);
    if (previousOccluder != nextOccluder) {
        const std::size_t column = static_cast<std::size_t>(x);
        if (nextOccluder) {
            skyOccluderY_[column] =
                std::min(skyOccluderY_[column], y);
        } else if (y <= skyOccluderY_[column]) {
            skyColumnDirty_[column] = 1;
        }
        ++solidRevision_;
    }
    if (material != previous) {
        std::uint8_t activityMask =
            materialActivityMask(previous) |
            materialActivityMask(material);
        if (isSkyOccluder(previous) !=
            isSkyOccluder(material)) {
            // Explicit construction or destruction can change support,
            // containment, gas paths, and heat conduction simultaneously.
            activityMask |= allMaterialActivity;
        }
        markMaterialActive(x, y, activityMask);
    }
    if (material != previous || material != Material::wood) {
        burnProgress_.set(index, 0.0F);
    }
    if (isLiquid(material)) {
        if (previous != material || liquidAmount_[index] == 0) {
            liquidAmount_.set(index, maximumLiquidMass);
        }
        if (previous != material) {
            liquidHeadDepth_.set(index, 1);
            liquidFoam_.set(index, 0);
        }
    } else if (material == Material::steam) {
        if (previous != Material::water && previous != Material::steam) {
            liquidAmount_.set(index, maximumLiquidMass);
        }
        liquidFlowX_.set(index, 0);
        liquidFlowY_.set(index, -127);
        liquidHeadDepth_.set(index, 0);
        liquidFoam_.set(index, 0);
    } else {
        liquidAmount_.set(index, 0);
        liquidFlowX_.set(index, 0);
        liquidFlowY_.set(index, 0);
        liquidHeadDepth_.set(index, 0);
        liquidFoam_.set(index, 0);
    }
    if (material == Material::fire) {
        heat_.set(index, 1.0F);
    } else if (material == Material::smoke) {
        heat_[index] = std::max(heat_[index], 0.12F);
        if (previous != Material::smoke) {
            const int variation = (x * 37 + y * 17) & 7;
            gasLifetime_[index] = 10.0F + static_cast<float>(variation) * 0.8F;
            gasDrift_[index] = ((x * 13 + y * 29) & 1) == 0 ? -1 : 1;
        }
    } else if (material == Material::steam) {
        if (previous != Material::steam) {
            gasDrift_[index] = ((x * 11 + y * 23) & 1) == 0 ? -1 : 1;
        }
    } else if (material == Material::air) {
        heat_.set(index, 0.0F);
    }
    if (material != Material::smoke) {
        gasLifetime_.set(index, 0.0F);
    }
    if (material != Material::smoke && material != Material::steam) {
        gasDrift_.set(index, 0);
    }
    if (material != Material::sand) {
        granularVelocityY_.set(index, 0.0F);
        granularFallRemainder_.set(index, 0.0F);
    } else if (previous != Material::sand) {
        granularVelocityY_.set(index, 0.0F);
        granularFallRemainder_.set(index, 0.0F);
    }
}

void World::swapCells(int firstX, int firstY, int secondX, int secondY) {
    const std::size_t first = indexOf(firstX, firstY);
    const std::size_t second = indexOf(secondX, secondY);
    const Material firstMaterial = cells_[first];
    const Material secondMaterial = cells_[second];
    const bool firstWasOccluder =
        isSkyOccluder(firstMaterial);
    const bool secondWasOccluder =
        isSkyOccluder(secondMaterial);
    if (firstWasOccluder != secondWasOccluder) {
        const auto updateColumn =
            [&](int x, int y, bool wasOccluder,
                bool isOccluderNow) {
                const std::size_t column =
                    static_cast<std::size_t>(x);
                if (isOccluderNow) {
                    skyOccluderY_[column] =
                        std::min(skyOccluderY_[column], y);
                } else if (wasOccluder &&
                           y <= skyOccluderY_[column]) {
                    skyColumnDirty_[column] = 1;
                }
            };
        updateColumn(firstX, firstY, firstWasOccluder,
                     secondWasOccluder);
        updateColumn(secondX, secondY, secondWasOccluder,
                     firstWasOccluder);
        ++solidRevision_;
    }
    std::swap(cells_[first], cells_[second]);
    std::swap(liquidAmount_[first], liquidAmount_[second]);
    std::swap(liquidFlowX_[first], liquidFlowX_[second]);
    std::swap(liquidFlowY_[first], liquidFlowY_[second]);
    std::swap(liquidHeadDepth_[first], liquidHeadDepth_[second]);
    std::swap(liquidFoam_[first], liquidFoam_[second]);
    std::swap(heat_[first], heat_[second]);
    std::swap(burnProgress_[first], burnProgress_[second]);
    std::swap(gasLifetime_[first], gasLifetime_[second]);
    std::swap(gasDrift_[first], gasDrift_[second]);
    std::swap(granularVelocityY_[first], granularVelocityY_[second]);
    std::swap(granularFallRemainder_[first],
              granularFallRemainder_[second]);
    std::uint8_t activityMask =
        materialActivityMask(firstMaterial) |
        materialActivityMask(secondMaterial);
    if (firstWasOccluder != secondWasOccluder) {
        activityMask |= granularActivity |
                        liquidActivity |
                        gasActivity;
    }
    if (heat_[first] > 0.015F || heat_[second] > 0.015F) {
        activityMask |= thermalActivity;
    }
    markMaterialActive(firstX, firstY, activityMask);
    markMaterialActive(secondX, secondY, activityMask);
}

std::uint8_t World::materialActivityMask(Material material) {
    switch (material) {
    case Material::sand:
        return granularActivity;
    case Material::water:
        return liquidActivity | thermalActivity;
    case Material::oil:
        return liquidActivity | thermalActivity;
    case Material::fire:
        return gasActivity | thermalActivity;
    case Material::smoke:
    case Material::steam:
        return gasActivity | thermalActivity;
    case Material::wood:
        return thermalActivity;
    default:
        return 0;
    }
}

void World::markMaterialActive(
    int x, int y, std::uint8_t activityMask) {
    if (activityMask == 0) {
        return;
    }
    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    // Wake the touched microtile and a one-tile halo. Sampling one tile away
    // naturally crosses a chunk boundary only when the source is close
    // enough to affect it, unlike the old unconditional 3x3 chunk wake.
    for (int offsetY = -1; offsetY <= 1; ++offsetY) {
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            const int wakeX = std::clamp(
                x + offsetX * materialMicrotileSize,
                0, width - 1);
            const int wakeY = std::clamp(
                y + offsetY * materialMicrotileSize,
                0, height - 1);
            const int chunkX = wakeX / chunkSize;
            const int chunkY = wakeY / chunkSize;
            const std::size_t chunkIndex =
                static_cast<std::size_t>(
                    chunkY * chunkColumns + chunkX);
            const int localX = wakeX - chunkX * chunkSize;
            const int localY = wakeY - chunkY * chunkSize;
            const int microtileX =
                localX / materialMicrotileSize;
            const int microtileY =
                localY / materialMicrotileSize;
            const std::uint64_t microtileBit =
                std::uint64_t{1}
                << static_cast<unsigned int>(
                       microtileY *
                           materialMicrotilesPerAxis +
                       microtileX);
            auto& activity =
                materialChunkActivity_[chunkIndex];
            for (std::size_t system = 0;
                 system < materialActivitySystemCount; ++system) {
                const std::uint8_t systemMask =
                    static_cast<std::uint8_t>(1U << system);
                if ((activityMask & systemMask) != 0) {
                    activity.lifetime[system] =
                        std::max<std::uint8_t>(
                            activity.lifetime[system], 8);
                    activity.microtiles[system] |=
                        microtileBit;
                }
            }
        }
    }
}

bool World::materialChunkActive(
    int x, int y, std::uint8_t activityMask) const {
    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return false;
    }
    const std::size_t chunkIndex = static_cast<std::size_t>(
        (y / chunkSize) * chunkColumns + x / chunkSize);
    const auto& activity = materialChunkActivity_[chunkIndex];
    for (std::size_t system = 0;
         system < materialActivitySystemCount; ++system) {
        const std::uint8_t systemMask =
            static_cast<std::uint8_t>(1U << system);
        if ((activityMask & systemMask) != 0 &&
            activity.lifetime[system] != 0) {
            return true;
        }
    }
    return false;
}

bool World::materialMicrotileActive(
    int x, int y, std::uint8_t activityMask) const {
    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return false;
    }
    const int chunkX = x / chunkSize;
    const int chunkY = y / chunkSize;
    const int microtileX =
        (x - chunkX * chunkSize) /
        materialMicrotileSize;
    const int microtileY =
        (y - chunkY * chunkSize) /
        materialMicrotileSize;
    const std::uint64_t microtileBit =
        std::uint64_t{1}
        << static_cast<unsigned int>(
               microtileY * materialMicrotilesPerAxis +
               microtileX);
    const auto& activity =
        materialChunkActivity_[static_cast<std::size_t>(
            chunkY * chunkColumns + chunkX)];
    for (std::size_t system = 0;
         system < materialActivitySystemCount; ++system) {
        const std::uint8_t systemMask =
            static_cast<std::uint8_t>(1U << system);
        if ((activityMask & systemMask) != 0 &&
            activity.lifetime[system] != 0 &&
            (activity.microtiles[system] &
             microtileBit) != 0) {
            return true;
        }
    }
    return false;
}

void World::ageMaterialChunks() {
    for (MaterialChunkActivity& activity :
         materialChunkActivity_) {
        for (std::size_t system = 0;
             system < materialActivitySystemCount; ++system) {
            std::uint8_t& lifetime =
                activity.lifetime[system];
            if (lifetime > 0) {
                --lifetime;
                if (lifetime == 0) {
                    activity.microtiles[system] = 0;
                }
            }
        }
    }
}

#ifdef GUNPOWDER_TEST_SCALE
void World::clearMaterialActivityForTest() {
    std::fill(materialChunkActivity_.begin(),
              materialChunkActivity_.end(),
              MaterialChunkActivity{});
}

std::array<bool, 4> World::materialActivityForTest(
    int x, int y) const {
    return {
        materialChunkActive(x, y, granularActivity),
        materialChunkActive(x, y, liquidActivity),
        materialChunkActive(x, y, gasActivity),
        materialChunkActive(x, y, thermalActivity),
    };
}

std::array<bool, 4> World::materialMicrotileActivityForTest(
    int x, int y) const {
    return {
        materialMicrotileActive(x, y, granularActivity),
        materialMicrotileActive(x, y, liquidActivity),
        materialMicrotileActive(x, y, gasActivity),
        materialMicrotileActive(x, y, thermalActivity),
    };
}
#endif

void World::rebuildDirtySkyColumns() {
    for (int x = 0; x < width; ++x) {
        const std::size_t column =
            static_cast<std::size_t>(x);
        if (skyColumnDirty_[column] == 0) {
            continue;
        }
        const int proceduralSurface = proceduralSurfaceY(x);
        int occluder = proceduralSurface;
        const int scanStart = std::clamp(
            skyOccluderY_[column], 0, proceduralSurface);
        const int scanEnd =
            std::min(height, proceduralSurface + chunkSize);
        for (int y = scanStart; y < scanEnd; ++y) {
            if (isSkyOccluder(cells_[indexOf(x, y)])) {
                occluder = y;
                break;
            }
        }
        skyOccluderY_[column] = occluder;
        skyColumnDirty_[column] = 0;
    }
}

void World::captureInteriorBackdrop() {
    // This is deliberately captured once after generation. Destructible
    // foreground terrain may subsequently open a path for sunlight, but it
    // must not turn the tomb's persistent interior backdrop into outdoor sky.
    const int chunksWide =
        (width + chunkSize - 1) / chunkSize;
    const int chunksHigh =
        (height + chunkSize - 1) / chunkSize;
    for (int chunkY = 0; chunkY < chunksHigh; ++chunkY) {
        for (int chunkX = 0; chunkX < chunksWide; ++chunkX) {
            const std::size_t chunkIndex =
                static_cast<std::size_t>(
                    chunkY * chunksWide + chunkX);
            if (generatedTerrainChunks_[chunkIndex] == 0) {
                continue;
            }
            const int beginX = chunkX * chunkSize;
            const int endX = std::min(width, beginX + chunkSize);
            const int beginY = chunkY * chunkSize;
            const int endY = std::min(height, beginY + chunkSize);
            for (int x = beginX; x < endX; ++x) {
                const int firstRoof =
                    skyOccluderY_[static_cast<std::size_t>(x)];
                for (int y = std::max(beginY, firstRoof);
                     y < endY; ++y) {
                    interiorBackdrop_.set(indexOf(x, y), 1);
                }
            }
        }
    }
}

void World::buildDirectionalSunHorizon(
    Vec2 direction, float samplesPerCell,
    std::vector<float>& depths,
    std::vector<std::int32_t>& blockers,
    float& minimumPerpendicularCoordinate,
    Vec2 receiverMinimum, Vec2 receiverMaximum) const {
    const float directionLength = length(direction);
    if (directionLength < 0.0001F) {
        depths.assign(1, -std::numeric_limits<float>::infinity());
        blockers.assign(1, -1);
        minimumPerpendicularCoordinate = 0.0F;
        return;
    }
    direction = direction * (1.0F / directionLength);
    samplesPerCell = std::max(samplesPerCell, 1.0F);

    const auto perpendicularCoordinate =
        [&](float x, float y) {
            return -direction.y * x + direction.x * y;
        };
    const std::array<float, 4> receiverCorners{
        perpendicularCoordinate(receiverMinimum.x, receiverMinimum.y),
        perpendicularCoordinate(receiverMaximum.x, receiverMinimum.y),
        perpendicularCoordinate(receiverMinimum.x, receiverMaximum.y),
        perpendicularCoordinate(
            receiverMaximum.x, receiverMaximum.y),
    };
    const auto [minimumCorner, maximumCorner] =
        std::minmax_element(receiverCorners.begin(),
                            receiverCorners.end());
    minimumPerpendicularCoordinate =
        *minimumCorner - samplesPerCell;
    const std::size_t sampleCount =
        static_cast<std::size_t>(
            std::ceil(
                (*maximumCorner - minimumPerpendicularCoordinate +
                 samplesPerCell) *
                samplesPerCell)) +
        2;
    depths.assign(
        sampleCount, -std::numeric_limits<float>::infinity());
    blockers.assign(sampleCount, -1);

    // A directional light collapses the world onto one axis perpendicular to
    // the rays. Rasterizing every solid's full projected square footprint
    // creates a conservative horizon with no diagonal gaps. Each bin stores
    // the exact exit depth where that quantized ray leaves the foremost solid;
    // using one approximate depth for an entire cell footprint produces long
    // alternating bands at shallow sun angles.
    const float projectedHalfExtent =
        0.5F * (std::abs(direction.x) +
                std::abs(direction.y));
    const bool hasHorizontalDirection =
        std::abs(direction.x) >= 0.00001F;
    const bool hasVerticalDirection =
        std::abs(direction.y) >= 0.00001F;
    const float inverseDirectionX =
        hasHorizontalDirection ? 1.0F / direction.x : 0.0F;
    const float inverseDirectionY =
        hasVerticalDirection ? 1.0F / direction.y : 0.0F;
    const int sunwardX =
        direction.x > 0.00001F
            ? 1
            : (direction.x < -0.00001F ? -1 : 0);
    const int sunwardY =
        direction.y > 0.00001F
            ? 1
            : (direction.y < -0.00001F ? -1 : 0);
    const auto solidAt = [&](int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return false;
        }
        const int chunksWide =
            (width + chunkSize - 1) / chunkSize;
        const std::size_t chunkIndex =
            static_cast<std::size_t>(
                (y / chunkSize) * chunksWide +
                (x / chunkSize));
        const Material material =
            generatedTerrainChunks_[chunkIndex] != 0
                ? cells_[indexOf(x, y)]
                : proceduralMaterial(x, y);
        return isSkyOccluder(material);
    };
    const auto rasterizeCell = [&](int x, int y) {
            const std::size_t cellIndex = indexOf(x, y);
            if (!solidAt(x, y)) {
                return;
            }
            // Only cells on a sun-facing material boundary can contribute to
            // the foremost horizon. Skipping buried cells changes no shadow
            // silhouette and avoids rasterizing millions of redundant solid
            // footprints in dense terrain.
            const bool exposedHorizontally =
                sunwardX != 0 &&
                !solidAt(x + sunwardX, y);
            const bool exposedVertically =
                sunwardY != 0 &&
                !solidAt(x, y + sunwardY);
            const bool exposedDiagonally =
                sunwardX != 0 && sunwardY != 0 &&
                !solidAt(x + sunwardX, y + sunwardY);
            if (!exposedHorizontally &&
                !exposedVertically &&
                !exposedDiagonally) {
                return;
            }
            const float centerX = static_cast<float>(x) + 0.5F;
            const float centerY = static_cast<float>(y) + 0.5F;
            const float perpendicular =
                perpendicularCoordinate(centerX, centerY);
            const int firstSample = std::max(
                0, static_cast<int>(std::floor(
                       (perpendicular - projectedHalfExtent -
                        minimumPerpendicularCoordinate) *
                       samplesPerCell)));
            const int lastSample = std::min(
                static_cast<int>(sampleCount) - 1,
                static_cast<int>(std::floor(
                    (perpendicular + projectedHalfExtent -
                     minimumPerpendicularCoordinate) *
                    samplesPerCell)));
            for (int sample = firstSample;
                 sample <= lastSample; ++sample) {
                const std::size_t sampleIndex =
                    static_cast<std::size_t>(sample);
                const float samplePerpendicular =
                    minimumPerpendicularCoordinate +
                    (static_cast<float>(sample) + 0.5F) /
                        samplesPerCell;
                const float lineOriginX =
                    -direction.y * samplePerpendicular;
                const float lineOriginY =
                    direction.x * samplePerpendicular;
                float entryDepth =
                    -std::numeric_limits<float>::infinity();
                float exitDepth =
                    std::numeric_limits<float>::infinity();
                if (hasHorizontalDirection) {
                    float first =
                        (static_cast<float>(x) - lineOriginX) *
                        inverseDirectionX;
                    float second =
                        (static_cast<float>(x + 1) - lineOriginX) *
                        inverseDirectionX;
                    if (first > second) {
                        std::swap(first, second);
                    }
                    entryDepth = std::max(entryDepth, first);
                    exitDepth = std::min(exitDepth, second);
                } else if (lineOriginX < static_cast<float>(x) ||
                           lineOriginX >
                               static_cast<float>(x + 1)) {
                    continue;
                }
                if (hasVerticalDirection) {
                    float first =
                        (static_cast<float>(y) - lineOriginY) *
                        inverseDirectionY;
                    float second =
                        (static_cast<float>(y + 1) - lineOriginY) *
                        inverseDirectionY;
                    if (first > second) {
                        std::swap(first, second);
                    }
                    entryDepth = std::max(entryDepth, first);
                    exitDepth = std::min(exitDepth, second);
                } else if (lineOriginY < static_cast<float>(y) ||
                           lineOriginY >
                               static_cast<float>(y + 1)) {
                    continue;
                }
                if (exitDepth < entryDepth) {
                    continue;
                }
                if (exitDepth > depths[sampleIndex]) {
                    depths[sampleIndex] = exitDepth;
                    blockers[sampleIndex] =
                        static_cast<std::int32_t>(cellIndex);
                }
            }
    };

    cells_.forEachAllocatedChunk(
        [&](int chunkX, int chunkY,
            const SparseGrid<Material>::Chunk& chunk) {
            const float maximumPerpendicularCoordinate =
                minimumPerpendicularCoordinate +
                static_cast<float>(sampleCount) / samplesPerCell;
            const int chunkMinX =
                chunkX * SparseGrid<Material>::chunkSize;
            const int chunkMinY =
                chunkY * SparseGrid<Material>::chunkSize;
            const int chunkMaxX = std::min(
                width, chunkMinX + SparseGrid<Material>::chunkSize);
            const int chunkMaxY = std::min(
                height, chunkMinY + SparseGrid<Material>::chunkSize);
            for (int y = chunkMinY; y < chunkMaxY; ++y) {
                const int beginX = chunkMinX;
                const int rowCells = chunkMaxX - chunkMinX;
                const int endX = beginX + rowCells;
                const std::array<float, 4> segmentCorners{
                    perpendicularCoordinate(
                        static_cast<float>(beginX),
                        static_cast<float>(y)),
                    perpendicularCoordinate(
                        static_cast<float>(endX),
                        static_cast<float>(y)),
                    perpendicularCoordinate(
                        static_cast<float>(beginX),
                        static_cast<float>(y + 1)),
                    perpendicularCoordinate(
                        static_cast<float>(endX),
                        static_cast<float>(y + 1)),
                };
                const auto [segmentMinimum, segmentMaximum] =
                    std::minmax_element(
                        segmentCorners.begin(),
                        segmentCorners.end());
                if (*segmentMaximum + projectedHalfExtent >=
                        minimumPerpendicularCoordinate &&
                    *segmentMinimum - projectedHalfExtent <=
                        maximumPerpendicularCoordinate) {
                    const std::size_t localBegin =
                        static_cast<std::size_t>(
                            y - chunkMinY) *
                        static_cast<std::size_t>(
                            SparseGrid<Material>::chunkSize);
                    for (int offset = 0; offset < rowCells; ++offset) {
                        const int x = beginX + offset;
                        const float perpendicular =
                            perpendicularCoordinate(
                                static_cast<float>(x) + 0.5F,
                                static_cast<float>(y) + 0.5F);
                        if (perpendicular + projectedHalfExtent <
                                minimumPerpendicularCoordinate ||
                            perpendicular - projectedHalfExtent >
                                maximumPerpendicularCoordinate) {
                            continue;
                        }
                        if (isSkyOccluder(
                                chunk[localBegin +
                                      static_cast<std::size_t>(
                                          offset)])) {
                            rasterizeCell(x, y);
                        }
                    }
                }
            }
        });

    // Unvisited chunks still have a deterministic surface silhouette. Add
    // that single boundary rather than scanning or materializing the entire
    // deep world for every change to directional sunlight.
    const int chunksWide =
        (width + chunkSize - 1) / chunkSize;
    const float maximumPerpendicularCoordinate =
        minimumPerpendicularCoordinate +
        static_cast<float>(sampleCount) / samplesPerCell;
    for (int x = 0; x < width; ++x) {
        const int y = proceduralSurfaceY(x);
        const float perpendicular =
            perpendicularCoordinate(
                static_cast<float>(x) + 0.5F,
                static_cast<float>(y) + 0.5F);
        if (perpendicular + projectedHalfExtent <
                minimumPerpendicularCoordinate ||
            perpendicular - projectedHalfExtent >
                maximumPerpendicularCoordinate) {
            continue;
        }
        const std::size_t chunkIndex =
            static_cast<std::size_t>(
                (y / chunkSize) * chunksWide +
                (x / chunkSize));
        if (generatedTerrainChunks_[chunkIndex] == 0) {
            rasterizeCell(x, y);
        }
    }
}

void World::destroyCircle(Vec2 center, float radius) {
    const int minX = static_cast<int>(std::floor(center.x - radius));
    const int maxX = static_cast<int>(std::ceil(center.x + radius));
    const int minY = static_cast<int>(std::floor(center.y - radius));
    const int maxY = static_cast<int>(std::ceil(center.y + radius));
    const float radiusSquared = radius * radius;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            const float dx = static_cast<float>(x) + 0.5F - center.x;
            const float dy = static_cast<float>(y) + 0.5F - center.y;
            if (dx * dx + dy * dy <= radiusSquared &&
                cell(x, y) != Material::rock) {
                setCell(x, y, Material::air);
            }
        }
    }
}

void World::paintCircle(Vec2 center, float radius, Material material) {
    const int minX = static_cast<int>(std::floor(center.x - radius));
    const int maxX = static_cast<int>(std::ceil(center.x + radius));
    const int minY = static_cast<int>(std::floor(center.y - radius));
    const int maxY = static_cast<int>(std::ceil(center.y + radius));
    const float radiusSquared = radius * radius;
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            const float dx = static_cast<float>(x) + 0.5F - center.x;
            const float dy = static_cast<float>(y) + 0.5F - center.y;
            if (dx * dx + dy * dy <= radiusSquared &&
                cell(x, y) != Material::rock) {
                setCell(x, y, material);
            }
        }
    }
}

void World::explode(Vec2 center, float radius) {
    struct DisplacedCell {
        Vec2 position;
        Material material;
        std::uint16_t amount;
    };
    std::vector<DisplacedCell> displaced;
    const int minX = std::max(0, static_cast<int>(std::floor(center.x - radius)));
    const int maxX = std::min(width - 1,
                              static_cast<int>(std::ceil(center.x + radius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - radius)));
    const int maxY = std::min(height - 1,
                              static_cast<int>(std::ceil(center.y + radius)));
    const float radiusSquared = radius * radius;

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            const Vec2 offset{
                static_cast<float>(x) + 0.5F - center.x,
                static_cast<float>(y) + 0.5F - center.y,
            };
            const float distanceSquared =
                offset.x * offset.x + offset.y * offset.y;
            if (distanceSquared > radiusSquared) {
                continue;
            }

            const std::size_t index = indexOf(x, y);
            const float strength =
                1.0F - std::sqrt(distanceSquared) / std::max(radius, 0.01F);
            heat_[index] = std::max(heat_[index], strength);
            const Material material = cells_[index];
            if (material == Material::rock) {
                continue;
            }
            if (material == Material::stone && strength < 0.56F) {
                continue;
            }
            if (material == Material::metal && strength < 0.82F) {
                continue;
            }
            if (material == Material::oil && strength > 0.32F) {
                setCell(x, y, Material::fire);
            } else if (isLiquid(material)) {
                displaced.push_back({{static_cast<float>(x), static_cast<float>(y)},
                                     material, liquidAmount_[index]});
                setCell(x, y, Material::air);
            } else {
                setCell(x, y, Material::air);
            }

        }
    }

    std::uniform_real_distribution<float> spread(0.0F, scaled(6.0F));
    for (const DisplacedCell& item : displaced) {
        const Vec2 direction = normalized(item.position - center);
        for (int attempt = 0; attempt < 5; ++attempt) {
            const float distance = radius + scaled(1.0F) + spread(random_);
            const int targetX =
                static_cast<int>(std::round(center.x + direction.x * distance));
            const int targetY =
                static_cast<int>(std::round(center.y + direction.y * distance));
            if (cell(targetX, targetY) == Material::air) {
                setCell(targetX, targetY, item.material);
                liquidAmount_[indexOf(targetX, targetY)] = item.amount;
                break;
            }
        }
    }

    // Hot gas at the blast center gives nearby oil a reliable ignition source.
    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            const float dx = static_cast<float>(x) + 0.5F - center.x;
            const float dy = static_cast<float>(y) + 0.5F - center.y;
            const float distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < radiusSquared * 0.16F &&
                cell(x, y) == Material::air) {
                setCell(x, y, ((x + y) % 3 == 0) ? Material::fire
                                                  : Material::smoke);
            }
        }
    }

    spawnExplosionParticles(center, radius);
    shakeStrength_ = std::max(shakeStrength_, radius * 0.42F);
    explosionFlash_ = std::max(explosionFlash_, 0.34F);
}

void World::updatePlayer(float dt, const InputState& input) {
    const Exposure exposure = sampleExposure(player_.position, playerHalfSize);
    const float previousWater = player_.waterSubmersion;
    const float previousOil = player_.oilSubmersion;
    updatePlayerEnvironment(dt, input, exposure);

    const float submersion =
        std::clamp(exposure.water + exposure.oil, 0.0F, 1.0F);
    const float horizontalInput = static_cast<float>(input.moveRight) -
                                  static_cast<float>(input.moveLeft);
    if (horizontalInput > 0.0F) {
        player_.facing = 1.0F;
    } else if (horizontalInput < 0.0F) {
        player_.facing = -1.0F;
    }
    const auto applyAirControl = [&](float acceleration) {
        if (horizontalInput > 0.0F &&
            player_.velocity.x < maximumRunSpeed) {
            player_.velocity.x =
                std::min(maximumRunSpeed,
                         player_.velocity.x + acceleration * dt);
        } else if (horizontalInput < 0.0F &&
                   player_.velocity.x > -maximumRunSpeed) {
            player_.velocity.x =
                std::max(-maximumRunSpeed,
                         player_.velocity.x - acceleration * dt);
        }
    };
    const bool hanging = grapple_.active && grapple_.attached;
    if (hanging && submersion <= 0.15F) {
        // A taut rope turns gravity into pendulum motion. Do not apply the
        // normal mid-air braking here because it would erase that tangential
        // momentum. A/D supplies only a modest pumping force.
        applyAirControl(scaled(38.0F));
    } else if (!player_.onGround && submersion <= 0.15F) {
        // Airborne motion is ballistic. This is especially important on the
        // first frame after releasing a swing: no-input air control must not
        // behave like friction and erase the inherited horizontal velocity.
        // A/D can still steer gradually without cancelling existing momentum.
        applyAirControl(scaled(72.0F));
    } else {
        const float targetVelocity =
            horizontalInput *
            (submersion > 0.15F ? scaled(31.0F) : maximumRunSpeed);
        const float acceleration =
            player_.onGround ? scaled(340.0F)
                             : (submersion > 0.15F
                                    ? scaled(115.0F)
                                    : scaled(150.0F));
        const float delta = std::clamp(targetVelocity - player_.velocity.x,
                                       -acceleration * dt, acceleration * dt);
        player_.velocity.x += delta;
    }

    if (input.jump && player_.onGround) {
        player_.velocity.y = scaled(-58.0F);
        player_.onGround = false;
    }
    if (input.swimUp && submersion > 0.12F) {
        player_.velocity.y =
            std::max(player_.velocity.y -
                         scaled(82.0F) * submersion * dt,
                     scaled(-38.0F));
    }

    const float buoyancy =
        exposure.water * 1.26F + exposure.oil * 1.02F;
    player_.velocity.y =
        std::min(player_.velocity.y + gravity * (1.0F - buoyancy) * dt,
                 terminalFallSpeed);
    const float drag =
        std::exp(-(exposure.water * 4.4F + exposure.oil * 2.4F) * dt);
    player_.velocity *= drag;

    movePlayerWithCollisions(player_.velocity * dt);

    const Exposure movedExposure =
        sampleExposure(player_.position, playerHalfSize);
    const float movedSubmersion =
        std::clamp(movedExposure.water + movedExposure.oil, 0.0F, 1.0F);
    const float speed = length(player_.velocity);
    if ((previousWater < 0.08F && movedExposure.water > 0.15F) ||
        (previousOil < 0.08F && movedExposure.oil > 0.15F)) {
        const Material splashMaterial =
            movedExposure.water >= movedExposure.oil ? Material::water
                                                      : Material::oil;
        spawnSplashParticles(player_.position, splashMaterial,
                             std::clamp(speed / scaled(45.0F),
                                        0.35F, 1.0F),
                             player_.velocity);
    }
    playerSplashAccumulator_ += speed * movedSubmersion * dt;
    if (playerSplashAccumulator_ >= scaled(5.0F)) {
        playerSplashAccumulator_ -= scaled(5.0F);
        spawnSplashParticles(
            {player_.position.x, player_.position.y + playerHalfSize.y},
            movedExposure.water >= movedExposure.oil ? Material::water
                                                      : Material::oil,
            0.28F, player_.velocity);
    }
    if (movedSubmersion > 0.05F && speed > scaled(2.0F)) {
        playerLiquidDisplacementAccumulator_ +=
            speed * movedSubmersion * dt;
        const float displacementInterval = scaled(0.75F);
        if (playerLiquidDisplacementAccumulator_ >=
            displacementInterval) {
            playerLiquidDisplacementAccumulator_ -=
                displacementInterval;
            displaceLiquid(
                player_.position, playerHalfSize,
                player_.velocity * dt,
                std::clamp(
                    movedSubmersion *
                        (0.25F + speed / scaled(55.0F)),
                    0.0F, 1.0F));
        }
    } else {
        playerLiquidDisplacementAccumulator_ = 0.0F;
    }
}

void World::movePlayerWithCollisions(Vec2 displacement) {
    const float largestMovement =
        std::max(std::abs(displacement.x), std::abs(displacement.y));
    const int steps =
        std::max(1, static_cast<int>(
                        std::ceil(largestMovement / scaled(0.35F))));
    const Vec2 movement =
        displacement * (1.0F / static_cast<float>(steps));
    const bool startedGrounded = player_.onGround;
    player_.onGround = false;

    for (int step = 0; step < steps; ++step) {
        const Vec2 start = player_.position;
        const Vec2 fullCandidate = start + movement;
        if (!overlapsTerrain(fullCandidate, playerHalfSize)) {
            player_.position = fullCandidate;
            continue;
        }

        bool blockedX = std::abs(movement.x) > 0.00001F;
        bool blockedY = std::abs(movement.y) > 0.00001F;
        if (blockedX) {
            const Vec2 horizontalCandidate =
                player_.position + Vec2{movement.x, 0.0F};
            if (!overlapsTerrain(horizontalCandidate, playerHalfSize)) {
                player_.position = horizontalCandidate;
                blockedX = false;
            } else if ((startedGrounded || player_.onGround) &&
                       player_.velocity.y >= 0.0F &&
                       tryPlayerStepUp(movement.x, playerStepHeight)) {
                // Small voxel ledges are part of an uneven walking surface,
                // not walls. Lift only as far as needed and let the ground
                // snap below settle the rounded body onto the new surface.
                blockedX = false;
                blockedY = false;
            }
        }
        if (blockedY) {
            const Vec2 verticalCandidate =
                player_.position + Vec2{0.0F, movement.y};
            if (!overlapsTerrain(verticalCandidate, playerHalfSize)) {
                player_.position = verticalCandidate;
                blockedY = false;
            }
        }

        // A rounded body can still meet a voxel corner exactly. Search a very
        // small distance along the collision tangent so rope tension carries
        // the player smoothly past the ledge instead of pinning both axes.
        if (blockedX && blockedY && !startedGrounded &&
            !player_.onGround) {
            const int preferredDirection =
                player_.velocity.x < 0.0F ? -1 : 1;
            for (float distance :
                 {scaled(0.25F), scaled(0.50F), scaled(0.75F),
                  scaled(1.0F)}) {
                for (int direction : {preferredDirection,
                                      -preferredDirection}) {
                    const Vec2 candidate =
                        start + Vec2{static_cast<float>(direction) * distance,
                                     movement.y};
                    if (!overlapsTerrain(candidate, playerHalfSize)) {
                        player_.position = candidate;
                        blockedX = false;
                        blockedY = false;
                        break;
                    }
                }
                if (!blockedY) {
                    break;
                }
            }
        }
        if (blockedX && blockedY && !startedGrounded &&
            !player_.onGround) {
            const int preferredDirection =
                player_.velocity.y < 0.0F ? -1 : 1;
            for (float distance :
                 {scaled(0.25F), scaled(0.50F), scaled(0.75F)}) {
                for (int direction : {preferredDirection,
                                      -preferredDirection}) {
                    const Vec2 candidate =
                        start + Vec2{movement.x,
                                     static_cast<float>(direction) * distance};
                    if (!overlapsTerrain(candidate, playerHalfSize)) {
                        player_.position = candidate;
                        blockedX = false;
                        blockedY = false;
                        break;
                    }
                }
                if (!blockedX) {
                    break;
                }
            }
        }

        if (blockedX) {
            player_.velocity.x = 0.0F;
        }
        if (blockedY) {
            if (movement.y > 0.0F) {
                player_.onGround = true;
            }
            player_.velocity.y = 0.0F;
        }
    }

    // Gravity advances a standing player by only a fraction of a cell each
    // frame. Without a contact envelope, small downward changes in terrain
    // make grounded movement alternate between ground and air control. Follow
    // nearby ground while descending, but never pull a jumping or
    // rope-lifted player back down.
    if (!player_.onGround && startedGrounded &&
        player_.velocity.y >= 0.0F) {
        (void)snapPlayerToGround(playerGroundSnapDistance);
    }
}

bool World::tryPlayerStepUp(float horizontalMovement,
                            float maximumHeight) {
    const Vec2 start = player_.position;
    for (float lift = playerGroundProbeIncrement;
         lift <= maximumHeight + 0.0001F;
         lift += playerGroundProbeIncrement) {
        const Vec2 candidate =
            start + Vec2{horizontalMovement, -lift};
        if (overlapsTerrain(candidate, playerHalfSize)) {
            continue;
        }

        player_.position = candidate;
        if (!snapPlayerToGround(lift + playerGroundProbeIncrement)) {
            player_.position = start;
            continue;
        }
        if (player_.velocity.y > 0.0F) {
            player_.velocity.y = 0.0F;
        }
        return true;
    }
    return false;
}

bool World::snapPlayerToGround(float maximumDistance) {
    if (maximumDistance <= 0.0F ||
        overlapsTerrain(player_.position, playerHalfSize)) {
        return false;
    }

    float previousDistance = 0.0F;
    for (float distance = playerGroundProbeIncrement;
         distance <= maximumDistance + 0.0001F;
         distance += playerGroundProbeIncrement) {
        const float testedDistance = std::min(distance, maximumDistance);
        if (!overlapsTerrain(
                player_.position + Vec2{0.0F, testedDistance},
                playerHalfSize)) {
            previousDistance = testedDistance;
            continue;
        }

        // Refine the final free position so settling does not inherit the
        // probe increment as visible vertical jitter.
        float freeDistance = previousDistance;
        float blockedDistance = testedDistance;
        for (int refinement = 0; refinement < 7; ++refinement) {
            const float midpoint =
                (freeDistance + blockedDistance) * 0.5F;
            if (overlapsTerrain(
                    player_.position + Vec2{0.0F, midpoint},
                    playerHalfSize)) {
                blockedDistance = midpoint;
            } else {
                freeDistance = midpoint;
            }
        }
        player_.position.y += freeDistance;
        player_.onGround = true;
        if (player_.velocity.y > 0.0F) {
            player_.velocity.y = 0.0F;
        }
        return true;
    }
    return false;
}

World::Exposure World::sampleExposure(Vec2 center, Vec2 halfSize) const {
    Exposure exposure;
    const int left = std::max(0, static_cast<int>(
        std::floor(center.x - halfSize.x)));
    const int right = std::min(width - 1, static_cast<int>(
        std::floor(center.x + halfSize.x)));
    const int top = std::max(0, static_cast<int>(
        std::floor(center.y - halfSize.y)));
    const int bottom = std::min(height - 1, static_cast<int>(
        std::floor(center.y + halfSize.y)));
    int samples = 0;
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            const std::size_t index = indexOf(x, y);
            const Material material = cells_[index];
            const float liquidFraction = std::min(
                static_cast<float>(liquidAmount_[index]) /
                    static_cast<float>(maximumLiquidMass),
                1.0F);
            if (material == Material::water) {
                exposure.water += liquidFraction;
            } else if (material == Material::oil) {
                exposure.oil += liquidFraction;
            } else if (material == Material::fire) {
                exposure.fire += 1.0F;
            } else if (material == Material::smoke) {
                exposure.smoke += 1.0F;
            } else if (material == Material::steam) {
                exposure.smoke += 0.35F;
            }
            exposure.heat += heat_[index];
            ++samples;
        }
    }
    if (samples > 0) {
        const float inverse = 1.0F / static_cast<float>(samples);
        exposure.water *= inverse;
        exposure.oil *= inverse;
        exposure.fire *= inverse;
        exposure.smoke *= inverse;
        exposure.heat *= inverse;
    }
    return exposure;
}

void World::updatePlayerEnvironment(float dt, const InputState&,
                                    const Exposure& exposure) {
    player_.waterSubmersion = exposure.water;
    player_.oilSubmersion = exposure.oil;

    player_.wetness = std::clamp(
        player_.wetness +
            (exposure.water * 2.8F - (1.0F - exposure.water) * 0.055F -
             exposure.heat * 0.32F) *
                dt,
        0.0F, 1.0F);
    player_.oiliness = std::clamp(
        player_.oiliness +
            (exposure.oil * 1.8F - exposure.water * 0.75F - 0.012F) * dt,
        0.0F, 1.0F);

    const float ignition =
        exposure.fire * (0.55F + player_.oiliness * 1.8F) +
        std::max(0.0F, exposure.heat - 0.62F) * player_.oiliness;
    player_.burning = std::max(player_.burning, ignition);
    const float extinguishing = 0.09F + player_.wetness * 4.8F +
                                exposure.water * 7.0F;
    player_.burning =
        std::clamp(player_.burning - extinguishing * dt, 0.0F, 1.0F);

    player_.suffocation = std::clamp(
        player_.suffocation +
            (exposure.smoke * 1.7F - (1.0F - exposure.smoke) * 0.48F) * dt,
        0.0F, 1.0F);
    const float suffocationDamage =
        std::max(0.0F, player_.suffocation - 0.62F) * 15.0F;
    player_.health -=
        (exposure.fire * 23.0F + player_.burning * 10.0F +
         suffocationDamage) *
        dt;

    statusParticleAccumulator_ +=
        (player_.burning * 18.0F + player_.suffocation * 2.0F) * dt;
    while (statusParticleAccumulator_ >= 1.0F) {
        statusParticleAccumulator_ -= 1.0F;
        std::uniform_real_distribution<float> offsetX(-playerHalfSize.x,
                                                      playerHalfSize.x);
        std::uniform_real_distribution<float> offsetY(-playerHalfSize.y,
                                                      playerHalfSize.y);
        const bool ember = player_.burning > 0.08F;
        emitParticle({
            player_.position + Vec2{offsetX(random_), offsetY(random_)},
            {offsetX(random_) * 1.5F,
             ember ? scaled(-16.0F) : scaled(-7.0F)},
            ember ? std::array<float, 3>{1.0F, 0.25F, 0.02F}
                  : std::array<float, 3>{0.42F, 0.44F, 0.48F},
            0.42F,
            0.42F,
            ember ? scaled(0.65F) : scaled(0.9F),
        });
    }

    if (player_.health <= 0.0F) {
        spawnExplosionParticles(player_.position, scaled(5.0F));
        player_ = Player{
            .position = {
                scaled(48.0F) +
                    static_cast<float>(generationOffsetX_),
                scaled(80.0F) +
                    static_cast<float>(generationOffsetY_),
            },
            .velocity = {},
            .facing = 1.0F,
            .onGround = false,
        };
        releaseGrapple();
        playerSplashAccumulator_ = 0.0F;
        playerLiquidDisplacementAccumulator_ = 0.0F;
        statusParticleAccumulator_ = 0.0F;
    }
}

void World::updateGrapple(float dt, const InputState& input) {
    constexpr float hookSpeed = scaled(235.0F);
    constexpr float maximumRange = scaled(280.0F);

    if (input.grappleToggle) {
        if (grapple_.active) {
            releaseGrapple();
            return;
        }
    }

    if (!grapple_.active) {
        if (!input.grappleToggle) {
            return;
        }
        const Vec2 direction = normalized(input.aim - player_.position);
        grapple_.active = true;
        grapple_.attached = false;
        grapple_.hookPosition =
            player_.position + direction * scaled(5.0F);
        grapple_.hookVelocity = direction * hookSpeed;
        grapple_.ropeLength = 0.0F;
        grapple_.points = {player_.position, grapple_.hookPosition};
        grapple_.previousPoints = grapple_.points;
    }

    if (!grapple_.attached) {
        const Vec2 movement = grapple_.hookVelocity * dt;
        const int steps =
            std::max(1, static_cast<int>(std::ceil(length(movement))));
        for (int step = 0; step < steps && !grapple_.attached; ++step) {
            const Vec2 previous = grapple_.hookPosition;
            const Vec2 candidate =
                previous + movement * (1.0F / static_cast<float>(steps));
            const int cellX = static_cast<int>(std::floor(candidate.x));
            const int cellY = static_cast<int>(std::floor(candidate.y));
            if (isSolid(cellX, cellY)) {
                grapple_.attached = true;
                grapple_.anchor = previous;
                grapple_.hookPosition = previous;
                grapple_.anchorCellX = cellX;
                grapple_.anchorCellY = cellY;
                grapple_.ropeLength = std::clamp(
                    length(player_.position - grapple_.anchor) * 1.04F,
                    scaled(10.0F), maximumRange);
                initializeGrappleChain();

                std::uniform_real_distribution<float> spark(
                    scaled(-12.0F), scaled(12.0F));
                for (int index = 0; index < 8; ++index) {
                    emitParticle({
                        grapple_.anchor,
                        {spark(random_), spark(random_)},
                        {0.95F, 0.76F, 0.28F},
                        0.24F,
                        0.24F,
                        scaled(0.42F),
                    });
                }
            } else {
                grapple_.hookPosition = candidate;
            }
        }
        if (!grapple_.attached &&
            length(grapple_.hookPosition - player_.position) > maximumRange) {
            releaseGrapple();
        } else if (!grapple_.attached) {
            grapple_.points = {player_.position, grapple_.hookPosition};
            grapple_.previousPoints = grapple_.points;
        }
        return;
    }

    if (!isSolid(grapple_.anchorCellX, grapple_.anchorCellY)) {
        releaseGrapple();
        return;
    }

    const float reelDirection =
        static_cast<float>(input.reelOut) - static_cast<float>(input.reelIn);
    grapple_.ropeLength = std::clamp(
        grapple_.ropeLength + reelDirection * scaled(42.0F) * dt,
        scaled(8.0F), maximumRange);

    updateGrappleWraps(dt);

    float fixedPathLength = 0.0F;
    Vec2 previousPathPoint = grapple_.anchor;
    for (auto point = grapple_.wrapPoints.rbegin();
         point != grapple_.wrapPoints.rend(); ++point) {
        fixedPathLength += length(*point - previousPathPoint);
        previousPathPoint = *point;
    }

    // Reeling cannot pull an already wrapped portion of the rope through
    // terrain. Once the player rounds that corner and it unwraps, reeling can
    // continue normally.
    constexpr float minimumFreeSegment = scaled(3.0F);
    grapple_.ropeLength =
        std::max(grapple_.ropeLength, fixedPathLength + minimumFreeSegment);
    const float freeSegmentLength =
        std::max(minimumFreeSegment, grapple_.ropeLength - fixedPathLength);
    const Vec2 tensionPoint = grapple_.wrapPoints.empty()
                                  ? grapple_.anchor
                                  : grapple_.wrapPoints.front();

    const Vec2 fromAnchor = player_.position - tensionPoint;
    const float distance = length(fromAnchor);
    if (distance > 0.001F) {
        Vec2 radialDirection = fromAnchor * (1.0F / distance);
        if (distance > freeSegmentLength) {
            // The rope is inextensible: correct the position to its maximum
            // radius, then remove only velocity that points farther away from
            // the anchor. Tangential velocity (including the component
            // created by gravity) is deliberately preserved.
            const Vec2 constrainedPosition =
                tensionPoint + radialDirection * freeSegmentLength;
            if (!overlapsTerrain(constrainedPosition, playerHalfSize)) {
                player_.position = constrainedPosition;
                radialDirection =
                    normalized(player_.position - tensionPoint);
            } else {
                // When the exact radial projection meets a voxel corner, try
                // nearby points on the same rope arc. This preserves rope
                // length while allowing tangential swing motion to slide
                // around the obstacle.
                const int preferredDirection =
                    (-radialDirection.y * player_.velocity.x +
                     radialDirection.x * player_.velocity.y) < 0.0F
                        ? -1
                        : 1;
                bool foundArcPosition = false;
                for (float arcDistance :
                     {scaled(0.35F), scaled(0.70F), scaled(1.05F),
                      scaled(1.40F), scaled(1.75F)}) {
                    const float angle =
                        std::min(0.22F, arcDistance / freeSegmentLength);
                    for (int direction :
                         {preferredDirection, -preferredDirection}) {
                        const float signedAngle =
                            angle * static_cast<float>(direction);
                        const float cosine = std::cos(signedAngle);
                        const float sine = std::sin(signedAngle);
                        const Vec2 rotated{
                            radialDirection.x * cosine -
                                radialDirection.y * sine,
                            radialDirection.x * sine +
                                radialDirection.y * cosine,
                        };
                        const Vec2 candidate =
                            tensionPoint + rotated * freeSegmentLength;
                        if (!overlapsTerrain(candidate, playerHalfSize)) {
                            player_.position = candidate;
                            radialDirection = rotated;
                            foundArcPosition = true;
                            break;
                        }
                    }
                    if (foundArcPosition) {
                        break;
                    }
                }
            }
        }

        // Catch outward motion at the taut boundary before numerical error
        // can add energy or visibly stretch the rope.
        if (distance >= freeSegmentLength - scaled(0.05F)) {
            const float radialVelocity =
                dot(player_.velocity, radialDirection);
            if (radialVelocity > 0.0F) {
                player_.velocity -= radialDirection * radialVelocity;
            }
        }
    }
    simulateGrappleChain(dt);
}

void World::updateGrappleWraps(float dt) {
    grapple_.wrapCooldown = std::max(0.0F, grapple_.wrapCooldown - dt);
    if (grapple_.wrapCooldown > 0.0F) {
        return;
    }

    // If the player can see past the nearest corner, that corner no longer
    // carries tension and the rope unwraps from it.
    if (!grapple_.wrapPoints.empty()) {
        const Vec2 pointBeyond =
            grapple_.wrapPoints.size() > 1 ? grapple_.wrapPoints[1]
                                           : grapple_.anchor;
        if (ropeLineClear(player_.position, pointBeyond)) {
            grapple_.wrapPoints.erase(grapple_.wrapPoints.begin());
            grapple_.wrapCooldown = 0.045F;
            initializeGrappleChain();
            return;
        }
    }

    const Vec2 currentTensionPoint = grapple_.wrapPoints.empty()
                                         ? grapple_.anchor
                                         : grapple_.wrapPoints.front();
    if (ropeLineClear(player_.position, currentTensionPoint)) {
        return;
    }

    Vec2 newWrapPoint{};
    if (!findRopeWrapPoint(player_.position, currentTensionPoint,
                           newWrapPoint)) {
        return;
    }
    for (Vec2 existing : grapple_.wrapPoints) {
        if (length(existing - newWrapPoint) < scaled(0.7F)) {
            return;
        }
    }
    if (grapple_.wrapPoints.size() >= 32) {
        return;
    }

    grapple_.wrapPoints.insert(grapple_.wrapPoints.begin(), newWrapPoint);
    grapple_.wrapCooldown = 0.06F;
    initializeGrappleChain();
}

bool World::ropeLineClear(Vec2 from, Vec2 to) const {
    const Vec2 delta = to - from;
    const float distance = length(delta);
    if (distance < scaled(0.35F)) {
        return true;
    }

    const int steps =
        std::max(2, static_cast<int>(
                        std::ceil(distance * 4.0F / spatialScale)));
    for (int step = 1; step < steps; ++step) {
        const float t =
            static_cast<float>(step) / static_cast<float>(steps);
        // Leave a small clearance around the endpoints. Hook and corner
        // points intentionally sit very close to solid terrain.
        if (t < 0.012F || t > 0.988F) {
            continue;
        }
        const Vec2 sample = from + delta * t;
        if (isSolid(static_cast<int>(std::floor(sample.x)),
                    static_cast<int>(std::floor(sample.y)))) {
            return false;
        }
    }
    return true;
}

bool World::findRopeWrapPoint(Vec2 from, Vec2 to, Vec2& wrapPoint) const {
    const Vec2 delta = to - from;
    const float distance = length(delta);
    if (distance < scaled(0.5F)) {
        return false;
    }

    int hitX = 0;
    int hitY = 0;
    bool foundHit = false;
    const int steps =
        std::max(2, static_cast<int>(
                        std::ceil(distance * 5.0F / spatialScale)));
    for (int step = 1; step < steps; ++step) {
        const float t =
            static_cast<float>(step) / static_cast<float>(steps);
        const Vec2 sample = from + delta * t;
        const int x = static_cast<int>(std::floor(sample.x));
        const int y = static_cast<int>(std::floor(sample.y));
        if (isSolid(x, y)) {
            hitX = x;
            hitY = y;
            foundHit = true;
            break;
        }
    }
    if (!foundHit) {
        return false;
    }

    // Search the nearby cell vertices. Each vertex is tested from all four
    // air-side offsets; the shortest point visible from both ends is the
    // physical corner the rope should bend around.
    constexpr float clearance = 0.14F;
    constexpr std::array<Vec2, 4> offsets{{
        {-clearance, -clearance},
        {clearance, -clearance},
        {-clearance, clearance},
        {clearance, clearance},
    }};
    float bestPathLength = 1.0e9F;
    bool foundCorner = false;
    for (int vertexY = hitY - 1; vertexY <= hitY + 2; ++vertexY) {
        for (int vertexX = hitX - 1; vertexX <= hitX + 2; ++vertexX) {
            for (Vec2 offset : offsets) {
                const Vec2 candidate{
                    static_cast<float>(vertexX) + offset.x,
                    static_cast<float>(vertexY) + offset.y,
                };
                if (isSolid(static_cast<int>(std::floor(candidate.x)),
                            static_cast<int>(std::floor(candidate.y))) ||
                    length(candidate - from) < scaled(0.6F) ||
                    length(candidate - to) < scaled(0.6F) ||
                    !ropeLineClear(from, candidate) ||
                    !ropeLineClear(candidate, to)) {
                    continue;
                }
                const float pathLength =
                    length(candidate - from) + length(to - candidate);
                if (pathLength < bestPathLength) {
                    bestPathLength = pathLength;
                    wrapPoint = candidate;
                    foundCorner = true;
                }
            }
        }
    }
    return foundCorner;
}

void World::initializeGrappleChain() {
    std::vector<Vec2> route;
    route.reserve(grapple_.wrapPoints.size() + 2);
    route.push_back(player_.position);
    route.insert(route.end(), grapple_.wrapPoints.begin(),
                 grapple_.wrapPoints.end());
    route.push_back(grapple_.anchor);

    float routeLength = 0.0F;
    for (std::size_t index = 1; index < route.size(); ++index) {
        routeLength += length(route[index] - route[index - 1]);
    }
    const int pointCount = std::clamp(
        std::max(
            static_cast<int>(
                std::ceil(grapple_.ropeLength / scaled(4.0F))) + 1,
            static_cast<int>(grapple_.wrapPoints.size()) + 2),
        3, 72);
    grapple_.points.resize(static_cast<std::size_t>(pointCount));
    grapple_.previousPoints.resize(static_cast<std::size_t>(pointCount));
    const float slack = std::max(0.0F, grapple_.ropeLength - routeLength);
    std::size_t segment = 1;
    float segmentStartDistance = 0.0F;
    float segmentLength = length(route[1] - route[0]);
    for (int index = 0; index < pointCount; ++index) {
        const float t =
            static_cast<float>(index) / static_cast<float>(pointCount - 1);
        const float targetDistance = routeLength * t;
        while (segment + 1 < route.size() &&
               targetDistance > segmentStartDistance + segmentLength) {
            segmentStartDistance += segmentLength;
            ++segment;
            segmentLength = length(route[segment] - route[segment - 1]);
        }
        const float segmentT =
            segmentLength > 0.0001F
                ? std::clamp((targetDistance - segmentStartDistance) /
                                 segmentLength,
                             0.0F, 1.0F)
                : 0.0F;
        Vec2 position = route[segment - 1] +
                        (route[segment] - route[segment - 1]) * segmentT;
        position.y +=
            std::sin(t * 3.14159F) *
            std::min(scaled(18.0F), slack + scaled(1.5F));
        grapple_.points[static_cast<std::size_t>(index)] = position;
        grapple_.previousPoints[static_cast<std::size_t>(index)] = position;
    }
}

void World::simulateGrappleChain(float dt) {
    if (grapple_.points.size() < 3 ||
        grapple_.points.size() != grapple_.previousPoints.size()) {
        initializeGrappleChain();
    }

    const std::size_t last = grapple_.points.size() - 1;
    std::vector<std::uint8_t> pinned(grapple_.points.size(), 0);
    std::vector<Vec2> pinPositions(grapple_.points.size());
    pinned.front() = 1;
    pinPositions.front() = player_.position;
    pinned.back() = 1;
    pinPositions.back() = grapple_.anchor;

    float routeLength = 0.0F;
    Vec2 routePoint = player_.position;
    for (Vec2 wrapPoint : grapple_.wrapPoints) {
        routeLength += length(wrapPoint - routePoint);
        routePoint = wrapPoint;
    }
    routeLength += length(grapple_.anchor - routePoint);

    float cumulativeLength = 0.0F;
    routePoint = player_.position;
    std::size_t previousPin = 0;
    for (std::size_t wrapIndex = 0;
         wrapIndex < grapple_.wrapPoints.size(); ++wrapIndex) {
        const Vec2 wrapPoint = grapple_.wrapPoints[wrapIndex];
        cumulativeLength += length(wrapPoint - routePoint);
        routePoint = wrapPoint;
        const std::size_t remainingPins =
            grapple_.wrapPoints.size() - wrapIndex - 1;
        const std::size_t maximumIndex =
            last > remainingPins + 1 ? last - remainingPins - 1
                                     : previousPin + 1;
        const float routeFraction =
            routeLength > 0.001F ? cumulativeLength / routeLength : 0.0F;
        std::size_t pinIndex = static_cast<std::size_t>(std::lround(
            routeFraction * static_cast<float>(last)));
        pinIndex =
            std::clamp(pinIndex, previousPin + 1, maximumIndex);
        pinned[pinIndex] = 1;
        pinPositions[pinIndex] = wrapPoint;
        previousPin = pinIndex;
    }

    const auto applyPins = [&]() {
        for (std::size_t index = 0; index <= last; ++index) {
            if (pinned[index] != 0) {
                grapple_.points[index] = pinPositions[index];
            }
        }
    };
    applyPins();
    for (std::size_t index = 1; index < last; ++index) {
        if (pinned[index] != 0) {
            grapple_.previousPoints[index] = pinPositions[index];
            continue;
        }
        Vec2& point = grapple_.points[index];
        Vec2& previous = grapple_.previousPoints[index];
        const Vec2 velocity = (point - previous) * 0.992F;
        previous = point;
        point += velocity;
        point.y += gravity * 0.24F * dt * dt;
    }

    const float segmentLength =
        grapple_.ropeLength / static_cast<float>(last);
    constexpr std::array<Vec2, 8> collisionOffsets{{
        {-1.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, -1.0F}, {0.0F, 1.0F},
        {-1.0F, -1.0F}, {1.0F, -1.0F},
        {-1.0F, 1.0F}, {1.0F, 1.0F},
    }};
    for (int iteration = 0; iteration < 10; ++iteration) {
        applyPins();
        for (std::size_t index = 0; index < last; ++index) {
            Vec2& first = grapple_.points[index];
            Vec2& second = grapple_.points[index + 1];
            const Vec2 delta = second - first;
            const float distance = length(delta);
            if (distance <= 0.0001F) {
                continue;
            }
            const Vec2 correction =
                delta * ((distance - segmentLength) / distance);
            const bool firstPinned = pinned[index] != 0;
            const bool secondPinned = pinned[index + 1] != 0;
            if (firstPinned && secondPinned) {
                continue;
            }
            if (firstPinned) {
                second -= correction;
            } else if (secondPinned) {
                first += correction;
            } else {
                first += correction * 0.5F;
                second -= correction * 0.5F;
            }
        }

        for (std::size_t index = 1; index < last; ++index) {
            if (pinned[index] != 0) {
                continue;
            }
            Vec2& point = grapple_.points[index];
            if (!isSolid(static_cast<int>(std::floor(point.x)),
                         static_cast<int>(std::floor(point.y)))) {
                continue;
            }
            const Vec2 previous = grapple_.previousPoints[index];
            if (!isSolid(static_cast<int>(std::floor(previous.x)),
                         static_cast<int>(std::floor(previous.y)))) {
                point = previous;
                continue;
            }
            for (Vec2 offset : collisionOffsets) {
                const Vec2 candidate = point + offset;
                if (!isSolid(static_cast<int>(std::floor(candidate.x)),
                             static_cast<int>(std::floor(candidate.y)))) {
                    point = candidate;
                    break;
                }
            }
        }
    }
    applyPins();
    for (std::size_t index = 0; index <= last; ++index) {
        if (pinned[index] != 0) {
            grapple_.previousPoints[index] = pinPositions[index];
        }
    }
}

void World::releaseGrapple() {
    grapple_ = Grapple{};
}

void World::updateBullets(float dt) {
    for (Projectile& bullet : bullets_) {
        bullet.lifetime -= dt;
        const Vec2 movement = bullet.velocity * dt;
        const int steps = std::max(1, static_cast<int>(std::ceil(length(movement))));
        for (int step = 0; step < steps && bullet.lifetime > 0.0F; ++step) {
            bullet.position += movement * (1.0F / static_cast<float>(steps));
            const int x = static_cast<int>(bullet.position.x);
            const int y = static_cast<int>(bullet.position.y);
            const Material material = cell(x, y);

            if (material == Material::rock) {
                bullet.lifetime = 0.0F;
            } else if (material == Material::stone) {
                destroyCircle(bullet.position, scaled(0.85F));
                bullet.energy -= 0.62F;
                bullet.velocity *= 0.52F;
            } else if (material == Material::metal) {
                destroyCircle(bullet.position, scaled(0.45F));
                bullet.energy -= 0.84F;
                bullet.velocity *= 0.36F;
            } else if (material == Material::wood) {
                destroyCircle(bullet.position, scaled(1.65F));
                bullet.energy -= 0.20F;
                bullet.velocity *= 0.86F;
            } else if (material == Material::dirt) {
                destroyCircle(bullet.position, scaled(1.7F));
                bullet.energy -= 0.34F;
                bullet.velocity *= 0.78F;
            } else if (material == Material::sand) {
                destroyCircle(bullet.position, scaled(1.2F));
                bullet.energy -= 0.16F;
                bullet.velocity *= 0.88F;
            } else if (isLiquid(material)) {
                bullet.energy -= 0.012F;
                bullet.velocity *= 0.985F;
            }

            if (bullet.energy <= 0.0F) {
                bullet.lifetime = 0.0F;
            }
        }
    }
    std::erase_if(bullets_, [](const Projectile& bullet) {
        return bullet.lifetime <= 0.0F;
    });
}

void World::updateGrenades(float dt) {
    for (Projectile& grenade : grenades_) {
        grenade.lifetime -= dt;
        const Exposure exposure =
            sampleExposure(
                grenade.position, {scaled(1.2F), scaled(1.2F)});
        const float submersion =
            std::clamp(exposure.water + exposure.oil, 0.0F, 1.0F);
        const float buoyancy =
            exposure.water * 0.98F + exposure.oil * 0.78F;
        grenade.velocity.y += gravity * (0.72F - buoyancy) * dt;
        grenade.velocity *= std::exp(
            -(exposure.water * 3.8F + exposure.oil * 2.0F) * dt);

        Vec2 next = grenade.position;
        next.x += grenade.velocity.x * dt;
        if (!isSolid(static_cast<int>(next.x), static_cast<int>(next.y))) {
            grenade.position.x = next.x;
        } else {
            grenade.velocity.x *= -0.62F;
        }

        next = grenade.position;
        next.y += grenade.velocity.y * dt;
        if (!isSolid(static_cast<int>(next.x), static_cast<int>(next.y))) {
            grenade.position.y = next.y;
        } else {
            grenade.velocity.y *= -0.58F;
            grenade.velocity.x *= 0.82F;
        }

        if (grenade.submersion < 0.08F && submersion > 0.16F) {
            spawnSplashParticles(
                grenade.position,
                exposure.water >= exposure.oil ? Material::water
                                                : Material::oil,
                std::clamp(length(grenade.velocity) / scaled(70.0F),
                           0.3F, 0.85F),
                grenade.velocity);
        }
        grenade.submersion = submersion;
        if (submersion > 0.08F &&
            length(grenade.velocity) > scaled(2.0F)) {
            displaceLiquid(
                grenade.position, {scaled(1.2F), scaled(1.2F)},
                           grenade.velocity * dt,
                           std::clamp(submersion * 0.55F, 0.0F, 0.75F));
        }

        if (grenade.lifetime <= 0.0F) {
            explode(grenade.position, scaled(13.0F));
        }
    }
    std::erase_if(grenades_, [](const Projectile& grenade) {
        return grenade.lifetime <= 0.0F;
    });
}

void World::updateMaterials() {
    const auto granularBegin =
        std::chrono::steady_clock::now();
    const ActiveBounds bounds = activeBounds();
    currentLiquidSelectionMs_ = 0.0F;
    currentLiquidGravityMs_ = 0.0F;
    currentLiquidLateralMs_ = 0.0F;
    currentLiquidFrontierMs_ = 0.0F;
    currentLiquidEqualizationMs_ = 0.0F;
    currentLiquidCandidateVisits_ = 0;
    currentLiquidPreparationCellVisits_ = 0;
    currentLiquidHeadSummaryHits_ = 0;
    currentLiquidEqualizationSeedVisits_ = 0;
    currentEqualizedComponents_ = 0;
    currentEqualizedCells_ = 0;
    rebuildLiquidWorklist_ = true;
    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    const int firstChunkX = bounds.minX / chunkSize;
    const int finalChunkX =
        (bounds.maxX + chunkSize - 1) / chunkSize;
    const int firstChunkY = bounds.minY / chunkSize;
    const int finalChunkY =
        (bounds.maxY + chunkSize - 1) / chunkSize;
    std::uint32_t activeChunkCount = 0;
    std::uint32_t activeGranularChunkCount = 0;
    std::uint32_t activeLiquidChunkCount = 0;
    std::uint32_t activeGasChunkCount = 0;
    std::uint32_t activeThermalChunkCount = 0;
    std::uint32_t activeGranularMicrotileCount = 0;
    std::uint32_t activeLiquidMicrotileCount = 0;
    std::uint32_t activeGasMicrotileCount = 0;
    std::uint32_t activeThermalMicrotileCount = 0;
    constexpr std::size_t granularSystem =
        std::countr_zero(
            static_cast<unsigned int>(granularActivity));
    constexpr std::size_t liquidSystem =
        std::countr_zero(
            static_cast<unsigned int>(liquidActivity));
    constexpr std::size_t gasSystem =
        std::countr_zero(
            static_cast<unsigned int>(gasActivity));
    constexpr std::size_t thermalSystem =
        std::countr_zero(
            static_cast<unsigned int>(thermalActivity));
    for (int chunkY = firstChunkY; chunkY < finalChunkY; ++chunkY) {
        for (int chunkX = firstChunkX; chunkX < finalChunkX; ++chunkX) {
            const int sampleX = chunkX * chunkSize;
            const int sampleY = chunkY * chunkSize;
            const bool granular = materialChunkActive(
                sampleX, sampleY, granularActivity);
            const bool liquid = materialChunkActive(
                sampleX, sampleY, liquidActivity);
            const bool gas = materialChunkActive(
                sampleX, sampleY, gasActivity);
            const bool thermal = materialChunkActive(
                sampleX, sampleY, thermalActivity);
            activeChunkCount +=
                granular || liquid || gas || thermal ? 1U : 0U;
            activeGranularChunkCount += granular ? 1U : 0U;
            activeLiquidChunkCount += liquid ? 1U : 0U;
            activeGasChunkCount += gas ? 1U : 0U;
            activeThermalChunkCount += thermal ? 1U : 0U;
            const auto& activity =
                materialChunkActivity_[
                    static_cast<std::size_t>(
                        chunkY * chunkColumns + chunkX)];
            activeGranularMicrotileCount +=
                static_cast<std::uint32_t>(
                    std::popcount(
                        activity.microtiles[granularSystem]));
            activeLiquidMicrotileCount +=
                static_cast<std::uint32_t>(
                    std::popcount(
                        activity.microtiles[liquidSystem]));
            activeGasMicrotileCount +=
                static_cast<std::uint32_t>(
                    std::popcount(
                        activity.microtiles[gasSystem]));
            activeThermalMicrotileCount +=
                static_cast<std::uint32_t>(
                    std::popcount(
                        activity.microtiles[thermalSystem]));
        }
    }
    materialSimulationTimings_.activeChunks = activeChunkCount;
    materialSimulationTimings_.activeGranularChunks =
        activeGranularChunkCount;
    materialSimulationTimings_.activeLiquidChunks =
        activeLiquidChunkCount;
    materialSimulationTimings_.activeGasChunks =
        activeGasChunkCount;
    materialSimulationTimings_.activeThermalChunks =
        activeThermalChunkCount;
    materialSimulationTimings_.activeGranularMicrotiles =
        activeGranularMicrotileCount;
    materialSimulationTimings_.activeLiquidMicrotiles =
        activeLiquidMicrotileCount;
    materialSimulationTimings_.activeGasMicrotiles =
        activeGasMicrotileCount;
    materialSimulationTimings_.activeThermalMicrotiles =
        activeThermalMicrotileCount;
    for (int chunkY = firstChunkY; chunkY < finalChunkY; ++chunkY) {
        for (int chunkX = firstChunkX; chunkX < finalChunkX; ++chunkX) {
            if (!materialChunkActive(
                    chunkX * chunkSize, chunkY * chunkSize)) {
                continue;
            }
            const int beginX =
                std::max(bounds.minX, chunkX * chunkSize);
            const int endX = std::min(
                bounds.maxX, (chunkX + 1) * chunkSize);
            const int beginY =
                std::max(bounds.minY, chunkY * chunkSize);
            const int endY = std::min(
                bounds.maxY, (chunkY + 1) * chunkSize);
            for (int y = beginY; y < endY; ++y) {
                for (int microtileX = 0;
                     microtileX < materialMicrotilesPerAxis;
                     ++microtileX) {
                    const int tileOriginX =
                        chunkX * chunkSize +
                        microtileX *
                            materialMicrotileSize;
                    const int tileBeginX =
                        std::max(beginX, tileOriginX);
                    const int tileEndX = std::min(
                        endX,
                        tileOriginX +
                            materialMicrotileSize);
                    if (tileBeginX >= tileEndX ||
                        !materialMicrotileActive(
                            tileBeginX, y)) {
                        continue;
                    }
                    for (int x = tileBeginX;
                         x < tileEndX; ++x) {
                        moved_.set(indexOf(x, y), 0);
                    }
                }
            }
        }
    }
    materialScanRight_ = !materialScanRight_;
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_int_distribution<int> percent(0, 99);

    const auto moveCell = [&](int fromX, int fromY, int toX, int toY) {
        const std::size_t from = indexOf(fromX, fromY);
        const std::size_t to = indexOf(toX, toY);
        swapCells(fromX, fromY, toX, toY);
        moved_[from] = 1;
        moved_[to] = 1;
    };

    // Granular positions remain cell-exact, but their free-fall integration
    // uses the same acceleration and terminal speed as the player. Fractional
    // travel is retained between 30 Hz material ticks, while a fast grain may
    // cross several open cells in one tick without tunneling through them.
    for (int y = std::min(bounds.maxY - 1, height - 2); y >= bounds.minY; --y) {
      const int chunkY = y / chunkSize;
      const int chunkCount = finalChunkX - firstChunkX;
      for (int chunkStep = 0; chunkStep < chunkCount; ++chunkStep) {
        const int chunkX =
            materialScanRight_
                ? firstChunkX + chunkStep
                : finalChunkX - 1 - chunkStep;
        if (!materialChunkActive(chunkX * chunkSize, chunkY * chunkSize,
                                 granularActivity)) {
          continue;
        }
        const int chunkOriginX = chunkX * chunkSize;
        for (int tileStep = 0; tileStep < materialMicrotilesPerAxis;
             ++tileStep) {
          const int microtileX =
              materialScanRight_
                  ? tileStep
                  : materialMicrotilesPerAxis - 1 - tileStep;
          const int tileOriginX =
              chunkOriginX + microtileX * materialMicrotileSize;
          const int beginX = std::max(bounds.minX, tileOriginX);
          const int endX =
              std::min(bounds.maxX, tileOriginX + materialMicrotileSize);
          if (beginX >= endX ||
              !materialMicrotileActive(beginX, y, granularActivity)) {
            continue;
          }
          for (int localStep = 0; localStep < endX - beginX; ++localStep) {
            const int x =
                materialScanRight_
                    ? beginX + localStep
                    : endX - 1 - localStep;
          const std::size_t index = indexOf(x, y);
          if (moved_[index] != 0) {
            continue;
          }
          const Material material = cells_[index];
          if (material != Material::sand) {
            continue;
          }

          const auto canFallInto = [](Material target) {
            return target == Material::air || target == Material::fire ||
                   target == Material::smoke || target == Material::steam ||
                   isLiquid(target);
          };

          if (canFallInto(cell(x, y + 1))) {
            const float previousVelocity =
                std::max(granularVelocityY_[index], 0.0F);
            const float nextVelocity =
                std::min(previousVelocity + gravity * materialTimeStep,
                         terminalFallSpeed);
            const float integratedDistance =
                granularFallRemainder_[index] +
                (previousVelocity + nextVelocity) * 0.5F * materialTimeStep;
            const int requestedSteps =
                static_cast<int>(std::floor(integratedDistance));
            if (requestedSteps <= 0) {
              granularVelocityY_[index] = nextVelocity;
              granularFallRemainder_[index] = integratedDistance;
              markMaterialActive(x, y, granularActivity);
              continue;
            }

            int currentY = y;
            int completedSteps = 0;
            while (completedSteps < requestedSteps && currentY < height - 1 &&
                   canFallInto(cell(x, currentY + 1))) {
              moveCell(x, currentY, x, currentY + 1);
              ++currentY;
              ++completedSteps;
            }

            const std::size_t destination = indexOf(x, currentY);
            const bool landed = completedSteps < requestedSteps ||
                                currentY >= height - 1 ||
                                !canFallInto(cell(x, currentY + 1));
            if (landed) {
              granularVelocityY_[destination] = 0.0F;
              granularFallRemainder_[destination] = 0.0F;
            } else {
              granularVelocityY_[destination] = nextVelocity;
              granularFallRemainder_[destination] =
                  integratedDistance - static_cast<float>(completedSteps);
              markMaterialActive(x, currentY, granularActivity);
            }
            continue;
          }

          granularVelocityY_[index] = 0.0F;
          granularFallRemainder_[index] = 0.0F;
          const int direction = coin(random_) == 0 ? -1 : 1;
          if (canFallInto(cell(x + direction, y + 1))) {
            moveCell(x, y, x + direction, y + 1);
            const std::size_t destination = indexOf(x + direction, y + 1);
            granularVelocityY_[destination] = 0.0F;
            granularFallRemainder_[destination] = 0.0F;
            continue;
          }
          if (canFallInto(cell(x - direction, y + 1))) {
            moveCell(x, y, x - direction, y + 1);
            const std::size_t destination = indexOf(x - direction, y + 1);
            granularVelocityY_[destination] = 0.0F;
            granularFallRemainder_[destination] = 0.0F;
          }
          }
        }
      }
    }

    // Local cellular substeps let falling liquids travel several pixels per
    // material tick without introducing fractional cell volume. Water gets
    // two additional passes so it levels rapidly while oil retains visibly
    // higher viscosity.
    const auto granularEnd =
        std::chrono::steady_clock::now();
    const bool hasActiveLiquidChunks =
        activeLiquidChunkCount != 0;
    if (hasActiveLiquidChunks) {
        cacheLiquidColumnHeads(bounds);
        prepareLiquidEqualization(bounds);
    } else {
        liquidWorklist_.clear();
        liquidNextWorklist_.clear();
        liquidEqualizationMoves_.clear();
    }
    const auto liquidPreparationEnd =
        std::chrono::steady_clock::now();
    constexpr int liquidSubsteps = 4;
    constexpr int additionalWaterSubsteps = 2;
    if (hasActiveLiquidChunks) {
        for (int step = 0; step < liquidSubsteps; ++step) {
            updateLiquids(bounds);
            applyLiquidEqualizationPhase(bounds, step);
        }
        for (int step = 0; step < additionalWaterSubsteps; ++step) {
            updateLiquids(bounds, true);
            applyLiquidEqualizationPhase(
                bounds, liquidSubsteps + step);
        }
    }
    const auto liquidTransportEnd =
        std::chrono::steady_clock::now();

    // Hot gases rise after liquids settle, and oil touching fire becomes fuel.
    const int gasChunkCount = finalChunkX - firstChunkX;
    for (int y = std::max(bounds.minY, 1); y < bounds.maxY; ++y) {
      const int chunkY = y / chunkSize;
      const int rowChunkOffset = gasChunkCount > 1
                                     ? std::uniform_int_distribution<int>(
                                           0, gasChunkCount - 1)(random_)
                                     : 0;
      for (int chunkStep = 0; chunkStep < gasChunkCount; ++chunkStep) {
        const int chunkX =
            firstChunkX + (chunkStep + rowChunkOffset) % gasChunkCount;
        if (!materialChunkActive(chunkX * chunkSize, chunkY * chunkSize,
                                 gasActivity)) {
          continue;
        }
        const int chunkOriginX = chunkX * chunkSize;
        const int microtileOffset =
            std::uniform_int_distribution<int>(
                0, materialMicrotilesPerAxis - 1)(random_);
        for (int tileStep = 0;
             tileStep < materialMicrotilesPerAxis; ++tileStep) {
          const int microtileX =
              (tileStep + microtileOffset) %
              materialMicrotilesPerAxis;
          const int tileOriginX =
              chunkOriginX +
              microtileX * materialMicrotileSize;
          const int beginX =
              std::max(bounds.minX, tileOriginX);
          const int endX = std::min(
              bounds.maxX,
              tileOriginX + materialMicrotileSize);
          if (beginX >= endX ||
              !materialMicrotileActive(
                  beginX, y, gasActivity)) {
            continue;
          }
          const int rowWidth = endX - beginX;
          const int rowOffset =
              rowWidth > 1
                  ? std::uniform_int_distribution<int>(
                        0, rowWidth - 1)(random_)
                  : 0;
          for (int localStep = 0;
               localStep < rowWidth; ++localStep) {
            const int rowPosition =
                (localStep + rowOffset) % rowWidth;
            const int x =
                materialScanRight_
                    ? beginX + rowPosition
                    : endX - 1 - rowPosition;
          const std::size_t index = indexOf(x, y);
          if (moved_[index] != 0) {
            continue;
          }
          const Material material = cells_[index];
          if (material == Material::fire) {
            markMaterialActive(x, y, gasActivity | thermalActivity);
            heat_[index] = std::max(0.0F, heat_[index] - 0.045F);
            constexpr std::array<Vec2, 4> neighbors{{
                {-1.0F, 0.0F},
                {1.0F, 0.0F},
                {0.0F, -1.0F},
                {0.0F, 1.0F},
            }};
            bool touchesWater = false;
            for (Vec2 offset : neighbors) {
              const int neighborX = x + static_cast<int>(offset.x);
              const int neighborY = y + static_cast<int>(offset.y);
              const Material neighbor = cell(neighborX, neighborY);
              if (neighbor == Material::oil && percent(random_) < 48) {
                heat_[indexOf(neighborX, neighborY)] = 1.0F;
              } else if (neighbor == Material::wood && percent(random_) < 22) {
                heat_[indexOf(neighborX, neighborY)] = 1.0F;
              } else if (neighbor == Material::water) {
                touchesWater = true;
                heat_[indexOf(neighborX, neighborY)] *= 0.45F;
              }
            }
            if (touchesWater) {
              heat_[index] *= 0.55F;
            }
            if (heat_[index] < 0.10F) {
              setCell(x, y, Material::smoke);
              continue;
            }
            const int riseDirection = coin(random_) == 0 ? -1 : 1;
            if (cell(x, y - 1) == Material::air ||
                cell(x, y - 1) == Material::smoke) {
              moveCell(x, y, x, y - 1);
            } else if (cell(x + riseDirection, y - 1) == Material::air) {
              moveCell(x, y, x + riseDirection, y - 1);
            }
          } else if (material == Material::smoke ||
                     material == Material::steam) {
            markMaterialActive(x, y, gasActivity | thermalActivity);
            const bool steam = material == Material::steam;
            heat_[index] *= steam ? 0.92F : 0.975F;
            if (steam && heat_[index] < 0.055F) {
              setCell(x, y, Material::water);
              liquidAmount_[index] = maximumLiquidMass;
              continue;
            }
            if (!steam) {
              gasLifetime_[index] -= 1.0F / 30.0F;
              if (gasLifetime_[index] <= 0.0F) {
                setCell(x, y, Material::air);
                continue;
              }
            }

            int drift = static_cast<int>(gasDrift_[index]);
            if (drift == 0) {
              drift = coin(random_) == 0 ? -1 : 1;
              gasDrift_[index] = static_cast<std::int8_t>(drift);
            } else if (percent(random_) < (steam ? 4 : 7)) {
              // Small independent eddies keep neighboring gas cells
              // from locking into the same horizontal travel lane.
              drift = -drift;
              gasDrift_[index] = static_cast<std::int8_t>(drift);
            }

            const Material materialAbove = cell(x, y - 1);
            const bool blockedByCeiling = materialAbove != Material::air &&
                                          materialAbove != Material::fire &&
                                          materialAbove != Material::smoke &&
                                          materialAbove != Material::steam;
            const int riseChance = steam ? 88 : 64;
            if (percent(random_) < riseChance) {
              // Persistent diagonal preference breaks up narrow vertical
              // columns while still letting buoyancy dominate.
              if (percent(random_) < 48 &&
                  cell(x + drift, y - 1) == Material::air) {
                moveCell(x, y, x + drift, y - 1);
                continue;
              }
              if (cell(x, y - 1) == Material::air) {
                moveCell(x, y, x, y - 1);
                continue;
              }
              if (cell(x + drift, y - 1) == Material::air) {
                moveCell(x, y, x + drift, y - 1);
                continue;
              }
              if (cell(x - drift, y - 1) == Material::air) {
                gasDrift_[index] = static_cast<std::int8_t>(-drift);
                moveCell(x, y, x - drift, y - 1);
                continue;
              }
            }

            // Smoke fans out beneath ceilings and continues to meander
            // laterally while rising through open rooms.
            const int lateralChance =
                blockedByCeiling ? (steam ? 58 : 76) : (steam ? 12 : 24);
            if (percent(random_) < lateralChance) {
              if (cell(x + drift, y) == Material::air) {
                moveCell(x, y, x + drift, y);
                continue;
              }
              if (cell(x - drift, y) == Material::air) {
                gasDrift_[index] = static_cast<std::int8_t>(-drift);
                moveCell(x, y, x - drift, y);
                continue;
              }
              gasDrift_[index] = static_cast<std::int8_t>(-drift);
            }
          }
          }
        }
      }
    }
    const auto gasAndReactionEnd =
        std::chrono::steady_clock::now();
    smoothTiming(
        materialSimulationTimings_.granularMs,
        elapsedMilliseconds(granularBegin, granularEnd));
    smoothTiming(
        materialSimulationTimings_.liquidPreparationMs,
        elapsedMilliseconds(granularEnd, liquidPreparationEnd));
    smoothTiming(
        materialSimulationTimings_.liquidTransportMs,
        elapsedMilliseconds(
            liquidPreparationEnd, liquidTransportEnd));
    smoothTiming(
        materialSimulationTimings_.liquidSelectionMs,
        currentLiquidSelectionMs_);
    smoothTiming(
        materialSimulationTimings_.liquidGravityMs,
        currentLiquidGravityMs_);
    smoothTiming(
        materialSimulationTimings_.liquidLateralMs,
        currentLiquidLateralMs_);
    smoothTiming(
        materialSimulationTimings_.liquidFrontierMs,
        currentLiquidFrontierMs_);
    smoothTiming(
        materialSimulationTimings_.liquidEqualizationMs,
        currentLiquidEqualizationMs_);
    smoothTiming(
        materialSimulationTimings_.gasAndReactionMs,
        elapsedMilliseconds(
            liquidTransportEnd, gasAndReactionEnd));
    materialSimulationTimings_.liquidCandidateVisits =
        currentLiquidCandidateVisits_;
    materialSimulationTimings_.liquidPreparationCellVisits =
        currentLiquidPreparationCellVisits_;
    materialSimulationTimings_.liquidHeadSummaryHits =
        currentLiquidHeadSummaryHits_;
    materialSimulationTimings_.liquidEqualizationSeedVisits =
        currentLiquidEqualizationSeedVisits_;
    materialSimulationTimings_.equalizedComponents =
        currentEqualizedComponents_;
    materialSimulationTimings_.equalizedCells =
        currentEqualizedCells_;
}

void World::cacheLiquidColumnHeads(const ActiveBounds& bounds) {
    ++liquidPreparationGeneration_;
    if (liquidPreparationGeneration_ == 0) {
        for (auto& summary : liquidChunkColumnSummaries_) {
            if (summary) {
                summary->generation = 0;
            }
        }
        ++liquidPreparationGeneration_;
    }

    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    const int firstChunkX = bounds.minX / chunkSize;
    const int finalChunkX =
        (bounds.maxX + chunkSize - 1) / chunkSize;
    const int firstChunkY = bounds.minY / chunkSize;
    const int finalChunkY =
        (bounds.maxY + chunkSize - 1) / chunkSize;
    constexpr std::uint8_t maximumCachedHead = 255;
    constexpr std::uint8_t foamDecayPerTick = 9;
    const auto incrementVisitCount = [&] {
        if (currentLiquidPreparationCellVisits_ <
            std::numeric_limits<std::uint32_t>::max()) {
            ++currentLiquidPreparationCellVisits_;
        }
    };

    // Process only awake liquid chunks. A chunk consumes the bottom-column
    // summary produced by its awake neighbor above. When that neighbor is
    // sleeping, a bounded upward walk reconstructs the exact incoming head
    // without scanning the rest of the camera region.
    for (int chunkY = firstChunkY;
         chunkY < finalChunkY; ++chunkY) {
        for (int chunkX = firstChunkX;
             chunkX < finalChunkX; ++chunkX) {
            const int chunkOriginX = chunkX * chunkSize;
            const int chunkOriginY = chunkY * chunkSize;
            if (!materialChunkActive(
                    chunkOriginX, chunkOriginY,
                    liquidActivity)) {
                continue;
            }

            const std::size_t chunkIndex =
                static_cast<std::size_t>(
                    chunkY * chunkColumns + chunkX);
            auto& summary =
                liquidChunkColumnSummaries_[chunkIndex];
            if (!summary) {
                summary =
                    std::make_unique<LiquidChunkColumnSummary>();
            }

            const int beginX =
                std::max(bounds.minX, chunkOriginX);
            const int endX = std::min(
                bounds.maxX, chunkOriginX + chunkSize);
            const int beginY =
                std::max(bounds.minY, chunkOriginY);
            const int endY = std::min(
                bounds.maxY, chunkOriginY + chunkSize);

            const LiquidChunkColumnSummary* aboveSummary = nullptr;
            if (chunkY > firstChunkY &&
                beginY == chunkOriginY) {
                const std::size_t aboveIndex =
                    static_cast<std::size_t>(
                        (chunkY - 1) * chunkColumns + chunkX);
                const auto& above =
                    liquidChunkColumnSummaries_[aboveIndex];
                if (above &&
                    above->generation ==
                        liquidPreparationGeneration_) {
                    aboveSummary = above.get();
                }
            }

            for (int x = beginX; x < endX; ++x) {
                const std::size_t localX =
                    static_cast<std::size_t>(
                        x - chunkOriginX);
                Material cachedMaterial = Material::air;
                std::uint8_t cachedDepth = 0;
                if (aboveSummary) {
                    cachedMaterial =
                        aboveSummary->bottomMaterial[localX];
                    cachedDepth =
                        aboveSummary->bottomDepth[localX];
                    if (currentLiquidHeadSummaryHits_ <
                        std::numeric_limits<std::uint32_t>::max()) {
                        ++currentLiquidHeadSummaryHits_;
                    }
                } else if (beginY > bounds.minY) {
                    int scanY = beginY - 1;
                    cachedMaterial =
                        cells_[indexOf(x, scanY)];
                    incrementVisitCount();
                    if (isLiquid(cachedMaterial)) {
                        cachedDepth = 1;
                        while (cachedDepth <
                                   maximumCachedHead &&
                               scanY > bounds.minY &&
                               cells_[indexOf(x, scanY - 1)] ==
                                   cachedMaterial) {
                            --scanY;
                            ++cachedDepth;
                            incrementVisitCount();
                        }
                    } else {
                        cachedMaterial = Material::air;
                    }
                }

                for (int y = beginY; y < endY; ++y) {
                    const std::size_t index = indexOf(x, y);
                    const Material material = cells_[index];
                    incrementVisitCount();
                    if (isLiquid(material)) {
                        if (cachedMaterial == material) {
                            cachedDepth =
                                static_cast<std::uint8_t>(
                                    std::min<int>(
                                        maximumCachedHead,
                                        static_cast<int>(
                                            cachedDepth) +
                                            1));
                        } else {
                            cachedMaterial = material;
                            cachedDepth = 1;
                        }
                        liquidHeadDepth_[index] = cachedDepth;
                    } else {
                        cachedMaterial = Material::air;
                        cachedDepth = 0;
                        liquidHeadDepth_[index] = 0;
                    }

                    if (material != Material::water) {
                        liquidFoam_[index] = 0;
                    } else if (liquidFoam_[index] >
                               foamDecayPerTick) {
                        liquidFoam_[index] =
                            static_cast<std::uint8_t>(
                                liquidFoam_[index] -
                                foamDecayPerTick);
                    } else {
                        liquidFoam_[index] = 0;
                    }
                }

                summary->bottomMaterial[localX] =
                    cachedMaterial;
                summary->bottomDepth[localX] =
                    cachedDepth;
            }
            summary->generation =
                liquidPreparationGeneration_;
        }
    }
}

void World::updateLiquids(const ActiveBounds& bounds, bool waterOnly) {
    const auto selectionBegin =
        std::chrono::steady_clock::now();
    std::uniform_int_distribution<int> coin(0, 1);
    std::uniform_int_distribution<int> percent(0, 99);
    const auto isOpen = [](Material material) {
        return material == Material::air || material == Material::fire ||
               material == Material::smoke || material == Material::steam;
    };
    const auto isCandidate = [&](std::size_t index) {
        const int x = static_cast<int>(
            index % static_cast<std::size_t>(width));
        const int y = static_cast<int>(
            index / static_cast<std::size_t>(width));
        if (x < bounds.minX || x >= bounds.maxX ||
            y < bounds.minY || y >= bounds.maxY ||
            y >= height - 1 ||
            !materialMicrotileActive(
                x, y, liquidActivity)) {
            return false;
        }
        const Material material = cells_[index];
        if (!isLiquid(material) ||
            liquidEqualizationReservation_[index] != 0) {
            return false;
        }
        const bool freeSurface =
            y <= bounds.minY ||
            cell(x, y - 1) != material ||
            cell(x - 1, y - 1) != material ||
            cell(x + 1, y - 1) != material;
        const bool canDescend =
            isOpen(cell(x, y + 1)) ||
            isOpen(cell(x - 1, y + 1)) ||
            isOpen(cell(x + 1, y + 1));
        const bool lateralBoundary =
            isOpen(cell(x - 1, y)) ||
            isOpen(cell(x + 1, y));
        const bool densityBoundary =
            material == Material::water &&
            cell(x, y + 1) == Material::oil;
        const bool carriesMomentum =
            std::abs(static_cast<int>(liquidFlowX_[index])) > 10 ||
            std::abs(static_cast<int>(liquidFlowY_[index])) > 10;
        return freeSurface || canDescend || lateralBoundary ||
               densityBoundary || carriesMomentum ||
               liquidFoam_[index] != 0;
    };

    if (rebuildLiquidWorklist_) {
        // Gather once from active chunks at the beginning of the material
        // tick. Later substeps advance a local frontier instead of rescanning
        // every active chunk.
        const int firstChunkX = bounds.minX / chunkSize;
        const int finalChunkX =
            (bounds.maxX + chunkSize - 1) / chunkSize;
        liquidWorklist_.clear();
        for (int y = bounds.minY; y < bounds.maxY; ++y) {
            const std::size_t rowWorkBegin = liquidWorklist_.size();
            const int chunkY = y / chunkSize;
            for (int chunkX = firstChunkX; chunkX < finalChunkX;
                 ++chunkX) {
                if (!materialChunkActive(
                        chunkX * chunkSize,
                        chunkY * chunkSize,
                        liquidActivity)) {
                    continue;
                }
                const int chunkOriginX =
                    chunkX * chunkSize;
                for (int microtileX = 0;
                     microtileX < materialMicrotilesPerAxis;
                     ++microtileX) {
                    const int tileOriginX =
                        chunkOriginX +
                        microtileX *
                            materialMicrotileSize;
                    const int beginX = std::max(
                        bounds.minX, tileOriginX);
                    const int endX = std::min(
                        bounds.maxX,
                        tileOriginX +
                            materialMicrotileSize);
                    if (beginX >= endX ||
                        !materialMicrotileActive(
                            beginX, y, liquidActivity)) {
                        continue;
                    }
                    for (int x = beginX; x < endX; ++x) {
                        moved_.set(indexOf(x, y), 0);
                    }
                    if (y >= height - 1) {
                        continue;
                    }
                    for (int x = beginX; x < endX; ++x) {
                        const std::size_t index =
                            indexOf(x, y);
                        if (liquidEqualizationReservation_[index] ==
                                0 &&
                            isCandidate(index)) {
                            liquidWorklist_.push_back(index);
                        }
                    }
                }
            }
            std::shuffle(
                liquidWorklist_.begin() +
                    static_cast<std::ptrdiff_t>(rowWorkBegin),
                liquidWorklist_.end(), random_);
        }
        rebuildLiquidWorklist_ = false;
    } else {
        // Generation-stamped insertion keeps this frontier unique without a
        // sort. Clear movement markers for liquid cells and their open targets
        // before filtering the raw neighborhood to current candidates.
        for (std::size_t index : liquidWorklist_) {
            moved_.set(index, 0);
        }
        std::erase_if(
            liquidWorklist_,
            [&](std::size_t index) {
                return !isCandidate(index);
            });
    }
    currentLiquidCandidateVisits_ +=
        static_cast<std::uint32_t>(std::min<std::size_t>(
            liquidWorklist_.size(),
            std::numeric_limits<std::uint32_t>::max() -
                currentLiquidCandidateVisits_));

    const auto targetInBounds = [&](int x, int y) {
        if (x < bounds.minX || x >= bounds.maxX ||
            y < bounds.minY || y >= bounds.maxY ||
            x < 0 || x >= width || y < 0 || y >= height) {
            return false;
        }
        return true;
    };
    ++liquidFrontierGeneration_;
    if (liquidFrontierGeneration_ == 0) {
        liquidFrontierStamp_.reset(0);
        ++liquidFrontierGeneration_;
    }
    liquidNextWorklist_.clear();
    const auto enqueueNext = [&](int x, int y) {
        if (!targetInBounds(x, y)) {
            return;
        }
        const std::size_t index = indexOf(x, y);
        if (liquidFrontierStamp_[index] ==
            liquidFrontierGeneration_) {
            return;
        }
        liquidFrontierStamp_[index] =
            liquidFrontierGeneration_;
        liquidNextWorklist_.push_back(index);
    };
    const auto canFallInto = [&](int x, int y) {
        if (!targetInBounds(x, y)) {
            return false;
        }
        const std::size_t target = indexOf(x, y);
        // A value of 2 marks a cell vacated by downward transport in this
        // pass. The next cell in the same stream may follow into it, keeping
        // one-cell-wide columns continuous.
        return liquidEqualizationReservation_[target] == 0 &&
               (moved_[target] == 0 || moved_[target] == 2) &&
               isOpen(cells_[target]);
    };
    const auto canMoveSidewaysInto = [&](int x, int y) {
        if (!targetInBounds(x, y)) {
            return false;
        }
        const std::size_t target = indexOf(x, y);
        return liquidEqualizationReservation_[target] == 0 &&
               moved_[target] == 0 && isOpen(cells_[target]);
    };
    const auto moveLiquid = [&](int fromX, int fromY, int toX, int toY,
                                 int flowX, int flowY) {
        const std::size_t from = indexOf(fromX, fromY);
        const std::size_t to = indexOf(toX, toY);
        const int previousFlowX =
            static_cast<int>(liquidFlowX_[from]);
        const int previousFlowY =
            static_cast<int>(liquidFlowY_[from]);
        const auto accumulateMomentum = [](int previous, int impulse) {
            if (impulse == 0) {
                return previous * 3 / 4;
            }
            const bool reversing =
                previous != 0 && ((previous < 0) != (impulse < 0));
            const int retained = reversing ? previous / 4
                                           : previous * 3 / 4;
            return std::clamp(retained + impulse * 3 / 4,
                              -127, 127);
        };
        swapCells(fromX, fromY, toX, toY);
        liquidAmount_[to] = maximumLiquidMass;
        liquidFlowX_[to] = static_cast<std::int8_t>(
            accumulateMomentum(previousFlowX, flowX));
        liquidFlowY_[to] = static_cast<std::int8_t>(
            accumulateMomentum(previousFlowY, flowY));
        moved_[from] = toY > fromY ? 2 : 1;
        moved_[to] = 1;
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                enqueueNext(
                    toX + offsetX, toY + offsetY);
            }
        }
    };
    const auto selectionEnd =
        std::chrono::steady_clock::now();
    currentLiquidSelectionMs_ +=
        elapsedMilliseconds(selectionBegin, selectionEnd);

    // The gathered rows are stored top-to-bottom, so reverse iteration is a
    // true bottom-up gravity pass. A stream can follow cells vacated earlier
    // in the same pass without imposing a left/right row pattern.
    for (auto iterator = liquidWorklist_.rbegin();
         iterator != liquidWorklist_.rend(); ++iterator) {
        const std::size_t source = *iterator;
        const int x = static_cast<int>(
            source % static_cast<std::size_t>(width));
        const int y = static_cast<int>(
            source / static_cast<std::size_t>(width));
        const Material material = cells_[source];
        if (!isLiquid(material) ||
            (waterOnly && material != Material::water) ||
            moved_[source] != 0 ||
            y >= height - 1) {
            continue;
        }
        liquidAmount_[source] = maximumLiquidMass;

        const std::size_t belowIndex = indexOf(x, y + 1);
        const Material below = cells_[belowIndex];
        if (material == Material::water &&
            below == Material::oil &&
            liquidEqualizationReservation_[belowIndex] == 0 &&
            moved_[belowIndex] != 1) {
            moveLiquid(x, y, x, y + 1, 0, 96);
            continue;
        }
        if (canFallInto(x, y + 1)) {
            moveLiquid(x, y, x, y + 1, 0, 127);
            continue;
        }

        const int firstDirection =
            liquidFlowX_[source] == 0
                ? (coin(random_) == 0 ? -1 : 1)
                : (liquidFlowX_[source] < 0 ? -1 : 1);
        for (int direction : {firstDirection, -firstDirection}) {
            if (canFallInto(x + direction, y + 1)) {
                moveLiquid(x, y, x + direction, y + 1,
                           direction * 84, 112);
                break;
            }
        }

        if (moved_[source] != 0) {
            continue;
        }

        // A fast falling water cell that meets support turns its downward
        // momentum into short-lived foam and an occasional spray particle.
        // The foam remains metadata; no material cells are created or lost.
        const int impactSpeed =
            static_cast<int>(liquidFlowY_[source]);
        const bool impactExposed =
            isOpen(cell(x - 1, y)) ||
            isOpen(cell(x + 1, y)) ||
            isOpen(cell(x, y - 1));
        if (material == Material::water &&
            impactSpeed > 72 && impactExposed) {
            liquidFoam_[source] = static_cast<std::uint8_t>(
                std::max<int>(liquidFoam_[source],
                              std::min(255, 72 + impactSpeed)));
            liquidFlowY_[source] = 0;
            if (impactSpeed > 104 && percent(random_) < 4) {
                const float direction =
                    coin(random_) == 0 ? -1.0F : 1.0F;
                emitParticle({
                    {static_cast<float>(x) + 0.5F,
                     static_cast<float>(y) + 0.2F},
                    {direction * scaled(8.0F),
                     -scaled(12.0F)},
                    {0.55F, 0.82F, 1.0F},
                    0.24F,
                    0.24F,
                    scaled(0.42F),
                });
            }
        } else {
            liquidFlowY_[source] = static_cast<std::int8_t>(
                impactSpeed * 2 / 3);
        }
    }
    const auto gravityEnd =
        std::chrono::steady_clock::now();
    currentLiquidGravityMs_ +=
        elapsedMilliseconds(selectionEnd, gravityEnd);

    // Surface relaxation remains randomized and destination-locked. This is
    // the part that must not cascade through newly vacated cells, because
    // doing so recreates horizontal shelves.
    std::shuffle(liquidWorklist_.begin(), liquidWorklist_.end(),
                 random_);
    for (std::size_t source : liquidWorklist_) {
        const int x = static_cast<int>(source %
                                       static_cast<std::size_t>(width));
        const int y = static_cast<int>(source /
                                       static_cast<std::size_t>(width));
        if (!materialMicrotileActive(
                x, y, liquidActivity)) {
            continue;
        }
        const Material material = cells_[source];
        if (!isLiquid(material) ||
            (waterOnly && material != Material::water) ||
            moved_[source] != 0) {
            continue;
        }
        liquidAmount_[source] = maximumLiquidMass;
        const int firstDirection =
            liquidFlowX_[source] == 0
                ? (coin(random_) == 0 ? -1 : 1)
                : (liquidFlowX_[source] < 0 ? -1 : 1);
        // A slope's boundary may still have liquid immediately above it.
        // Treat diagonal sky exposure as surface too; otherwise those cells
        // lock into a staircase and the pool freezes as a mound.
        const bool touchesFreeSurface =
            cell(x, y - 1) != material ||
            cell(x - 1, y - 1) != material ||
            cell(x + 1, y - 1) != material;
        const int headDepth =
            static_cast<int>(liquidHeadDepth_[source]);
        const bool hasLateralOutlet =
            isOpen(cell(x - 1, y)) ||
            isOpen(cell(x + 1, y));
        const int pressureThreshold =
            material == Material::water ? scaledCell(2)
                                        : scaledCell(5);
        const bool pressureDriven =
            hasLateralOutlet && headDepth >= pressureThreshold;
        if (!touchesFreeSurface && !pressureDriven) {
            liquidFlowX_[source] = static_cast<std::int8_t>(
                static_cast<int>(liquidFlowX_[source]) * 3 / 4);
            liquidFlowY_[source] = static_cast<std::int8_t>(
                static_cast<int>(liquidFlowY_[source]) * 2 / 3);
            continue;
        }

        const int baseSearchDistance =
            material == Material::water ? scaledCell(12)
                                        : scaledCell(3);
        const int pressureReach =
            material == Material::water
                ? std::min(headDepth, scaledCell(10))
                : std::min(headDepth / 3, scaledCell(2));
        const int momentumReach =
            std::abs(static_cast<int>(liquidFlowX_[source])) /
            (material == Material::water ? 8 : 18);
        const int searchDistance =
            baseSearchDistance + pressureReach + momentumReach;
        struct LateralPath {
            int direction = 0;
            int openRun = 0;
            int dropDistance = 0;
            int fallDepth = 0;
        };
        std::array<LateralPath, 2> paths{{
            {firstDirection, 0, 0, 0},
            {-firstDirection, 0, 0, 0},
        }};
        for (LateralPath& path : paths) {
            for (int distance = 1; distance <= searchDistance;
                 ++distance) {
                const int targetX = x + path.direction * distance;
                if (targetX < bounds.minX ||
                    targetX >= bounds.maxX) {
                    break;
                }
                const std::size_t target =
                    indexOf(targetX, y);
                if (liquidEqualizationReservation_[target] != 0 ||
                    !isOpen(cell(targetX, y))) {
                    break;
                }
                path.openRun = distance;
                if (isOpen(cell(targetX, y + 1))) {
                    path.dropDistance = distance;
                    const int maximumFallProbe =
                        material == Material::water
                            ? scaledCell(12)
                            : scaledCell(4);
                    for (int fall = 1;
                         fall <= maximumFallProbe &&
                         targetInBounds(targetX, y + fall) &&
                         isOpen(cell(targetX, y + fall));
                         ++fall) {
                        path.fallDepth = fall;
                    }
                    break;
                }
            }
        }

        const LateralPath* chosen = nullptr;
        const bool firstHasDrop = paths[0].dropDistance > 0;
        const bool secondHasDrop = paths[1].dropDistance > 0;
        if (firstHasDrop || secondHasDrop) {
            if (!firstHasDrop) {
                chosen = &paths[1];
            } else if (!secondHasDrop) {
                chosen = &paths[0];
            } else {
                // Prefer the side with the lower reachable liquid head.
                // Distance is only a tie-breaker, so a reservoir drains
                // toward the genuinely lower outlet instead of alternating.
                chosen =
                    paths[0].fallDepth != paths[1].fallDepth
                        ? (paths[0].fallDepth > paths[1].fallDepth
                               ? &paths[0]
                               : &paths[1])
                        : (paths[0].dropDistance <=
                                   paths[1].dropDistance
                               ? &paths[0]
                               : &paths[1]);
            }
        } else if (paths[0].openRun > 0 ||
                   paths[1].openRun > 0) {
            chosen = paths[0].openRun >= paths[1].openRun
                         ? &paths[0]
                         : &paths[1];
        }
        const int chosenDistance =
            chosen == nullptr
                ? 0
                : (chosen->dropDistance > 0
                       ? chosen->dropDistance
                       : chosen->openRun);
        if (chosenDistance > 0 &&
            canMoveSidewaysInto(
                x + chosen->direction * chosenDistance, y)) {
            const int direction = chosen->direction;
            const int pressureImpulse = std::min(
                127,
                (material == Material::water ? 82 : 45) +
                    headDepth *
                        (material == Material::water ? 3 : 1));
            moveLiquid(x, y,
                       x + direction * chosenDistance, y,
                       direction * pressureImpulse,
                       0);
            continue;
        }

        liquidFlowX_[source] = static_cast<std::int8_t>(
            static_cast<int>(liquidFlowX_[source]) * 2 / 3);
        liquidFlowY_[source] = 0;
    }
    const auto lateralEnd =
        std::chrono::steady_clock::now();
    currentLiquidLateralMs_ +=
        elapsedMilliseconds(gravityEnd, lateralEnd);

    // Every processed cell keeps its immediate neighborhood awake for the
    // next local substep. Far lateral destinations were added by moveLiquid.
    for (std::size_t source : liquidWorklist_) {
        const int sourceX = static_cast<int>(
            source % static_cast<std::size_t>(width));
        const int sourceY = static_cast<int>(
            source / static_cast<std::size_t>(width));
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                enqueueNext(
                    sourceX + offsetX,
                    sourceY + offsetY);
            }
        }
    }
    liquidWorklist_.swap(liquidNextWorklist_);
    currentLiquidFrontierMs_ += elapsedMilliseconds(
        lateralEnd, std::chrono::steady_clock::now());
}

void World::prepareLiquidEqualization(const ActiveBounds& bounds) {
    const auto isOpen = [](Material material) {
        return material == Material::air || material == Material::fire ||
               material == Material::smoke || material == Material::steam;
    };
    const auto insideBounds = [&](int x, int y) {
        return x >= bounds.minX && x < bounds.maxX && y >= bounds.minY &&
               y < bounds.maxY;
    };

    liquidEqualizationMoves_.clear();
    for (std::size_t index : liquidReservedCells_) {
        liquidEqualizationReservation_.set(index, 0);
    }
    liquidReservedCells_.clear();
    ++liquidComponentGeneration_;
    if (liquidComponentGeneration_ == 0) {
        liquidComponentStamp_.reset(0);
        ++liquidComponentGeneration_;
    }
    const auto reserveCell = [&](std::size_t index, std::uint8_t reservation) {
        if (liquidEqualizationReservation_[index] == 0) {
            liquidReservedCells_.push_back(index);
        }
        liquidEqualizationReservation_[index] = reservation;
    };

    constexpr std::array<std::array<int, 2>, 8> neighbors{{
        {{-1, 0}},
        {{1, 0}},
        {{0, -1}},
        {{0, 1}},
        {{-1, -1}},
        {{1, -1}},
        {{-1, 1}},
        {{1, 1}},
    }};
    const int firstChunkX = bounds.minX / chunkSize;
    const int finalChunkX = (bounds.maxX + chunkSize - 1) / chunkSize;
    for (int seedY = bounds.minY; seedY < bounds.maxY; ++seedY) {
        const int chunkY = seedY / chunkSize;
        for (int chunkX = firstChunkX; chunkX < finalChunkX; ++chunkX) {
            // A moving region wakes its own and neighboring chunks. Starting
            // discovery only there avoids rebuilding every settled lake each
            // tick. Once seeded, traversal still follows the complete
            // connected liquid component through sleeping chunks.
            if (!materialChunkActive(chunkX * chunkSize, chunkY * chunkSize,
                                     liquidActivity)) {
                continue;
            }
            const int chunkOriginX = chunkX * chunkSize;
            for (int microtileX = 0;
                 microtileX < materialMicrotilesPerAxis;
                 ++microtileX) {
                const int tileOriginX =
                    chunkOriginX +
                    microtileX * materialMicrotileSize;
                const int beginX =
                    std::max(bounds.minX, tileOriginX);
                const int endX = std::min(
                    bounds.maxX,
                    tileOriginX + materialMicrotileSize);
                if (beginX >= endX ||
                    !materialMicrotileActive(
                        beginX, seedY, liquidActivity)) {
                    continue;
                }
                for (int seedX = beginX; seedX < endX; ++seedX) {
                if (currentLiquidEqualizationSeedVisits_ <
                    std::numeric_limits<std::uint32_t>::max()) {
                    ++currentLiquidEqualizationSeedVisits_;
                }
                const std::size_t seed = indexOf(seedX, seedY);
                const Material material = cells_[seed];
                if (!isLiquid(material) ||
                    liquidComponentStamp_[seed] == liquidComponentGeneration_) {
                    continue;
                }

                liquidComponentQueue_.clear();
                liquidHighSurfaces_.clear();
                liquidLowSurfaces_.clear();
                liquidComponentQueue_.push_back(seed);
                liquidComponentStamp_[seed] = liquidComponentGeneration_;
                int highestSurfaceY = bounds.maxY;
                int lowestSurfaceY = bounds.minY - 1;
                bool hasDownwardPath = false;
                bool hasActiveFoam = false;
                bool hasDensityInstability = false;

                for (std::size_t cursor = 0;
                     cursor < liquidComponentQueue_.size(); ++cursor) {
                    const std::size_t current = liquidComponentQueue_[cursor];
                    const int x = static_cast<int>(
                        current % static_cast<std::size_t>(width));
                    const int y = static_cast<int>(
                        current / static_cast<std::size_t>(width));

                    const bool exposedAbove = isOpen(cell(x, y - 1));
                    const Material below = cell(x, y + 1);
                    hasDownwardPath = hasDownwardPath || isOpen(below) ||
                                      isOpen(cell(x - 1, y + 1)) ||
                                      isOpen(cell(x + 1, y + 1));
                    hasActiveFoam = hasActiveFoam || liquidFoam_[current] > 18;
                    hasDensityInstability = hasDensityInstability ||
                                            (material == Material::water &&
                                             below == Material::oil) ||
                                            (material == Material::oil &&
                                             cell(x, y - 1) == Material::water);
                    const bool supportedSurface =
                        exposedAbove && !isOpen(below);
                    if (supportedSurface) {
                        highestSurfaceY = std::min(highestSurfaceY, y);
                        lowestSurfaceY = std::max(lowestSurfaceY, y);
                    }

                    for (const auto& offset : neighbors) {
                        const int neighborX = x + offset[0];
                        const int neighborY = y + offset[1];
                        if (!insideBounds(neighborX, neighborY)) {
                            continue;
                        }
                        const std::size_t neighbor =
                            indexOf(neighborX, neighborY);
                        if (liquidComponentStamp_[neighbor] !=
                                liquidComponentGeneration_ &&
                            cells_[neighbor] == material) {
                            liquidComponentStamp_[neighbor] =
                                liquidComponentGeneration_;
                            liquidComponentQueue_.push_back(neighbor);
                        }
                    }
                }
                ++currentEqualizedComponents_;
                currentEqualizedCells_ +=
                    static_cast<std::uint32_t>(std::min<std::size_t>(
                        liquidComponentQueue_.size(),
                        std::numeric_limits<std::uint32_t>::max() -
                            currentEqualizedCells_));

                const bool hasSupportedSurface =
                    highestSurfaceY <= lowestSurfaceY;
                const int levelDifference =
                    hasSupportedSurface ? lowestSurfaceY - highestSurfaceY : 0;
                const bool settled = levelDifference <= 1 && !hasDownwardPath &&
                                     !hasActiveFoam && !hasDensityInstability;
                if (settled) {
                    // A discrete liquid may retain a partially occupied top
                    // row, but once its connected surfaces agree and nothing
                    // can fall, residual direction hints must not keep that row
                    // shuffling.
                    for (const std::size_t cellIndex : liquidComponentQueue_) {
                        liquidFlowX_[cellIndex] = 0;
                        liquidFlowY_[cellIndex] = 0;
                        reserveCell(cellIndex, 255);
                    }
                    continue;
                }
                if (levelDifference <= 1) {
                    continue;
                }

                for (const std::size_t surface : liquidComponentQueue_) {
                    const int x = static_cast<int>(
                        surface % static_cast<std::size_t>(width));
                    const int y = static_cast<int>(
                        surface / static_cast<std::size_t>(width));
                    if (!isOpen(cell(x, y - 1)) || isOpen(cell(x, y + 1))) {
                        continue;
                    }
                    if (y == highestSurfaceY) {
                        liquidHighSurfaces_.push_back(surface);
                    } else if (y == lowestSurfaceY) {
                        liquidLowSurfaces_.push_back(surface);
                    }
                }
                if (liquidHighSurfaces_.empty() || liquidLowSurfaces_.empty()) {
                    continue;
                }

                std::shuffle(liquidHighSurfaces_.begin(),
                             liquidHighSurfaces_.end(), random_);
                std::shuffle(liquidLowSurfaces_.begin(),
                             liquidLowSurfaces_.end(), random_);
                const std::size_t mobilityLimit = static_cast<std::size_t>(
                    material == Material::water ? scaledCell(16)
                                                : scaledCell(3));
                const std::size_t transferLimit = std::min({
                    mobilityLimit,
                    liquidHighSurfaces_.size(),
                    liquidLowSurfaces_.size(),
                    static_cast<std::size_t>(
                        std::max(1, levelDifference *
                                        (material == Material::water ? 2 : 1))),
                });

                std::size_t sourceCursor = 0;
                std::size_t destinationCursor = 0;
                std::size_t transfers = 0;
                while (transfers < transferLimit &&
                       sourceCursor < liquidHighSurfaces_.size() &&
                       destinationCursor < liquidLowSurfaces_.size()) {
                    const std::size_t source =
                        liquidHighSurfaces_[sourceCursor++];
                    const std::size_t lowerSurface =
                        liquidLowSurfaces_[destinationCursor++];
                    const int destinationX = static_cast<int>(
                        lowerSurface % static_cast<std::size_t>(width));
                    const int destinationY =
                        static_cast<int>(lowerSurface /
                                         static_cast<std::size_t>(width)) -
                        1;
                    if (!insideBounds(destinationX, destinationY)) {
                        continue;
                    }
                    const std::size_t destination =
                        indexOf(destinationX, destinationY);
                    const int phaseCount = material == Material::water ? 6 : 4;
                    const std::uint8_t phase = static_cast<std::uint8_t>(
                        transfers % static_cast<std::size_t>(phaseCount));
                    liquidEqualizationMoves_.push_back({
                        .source = source,
                        .destination = destination,
                        .support = lowerSurface,
                        .material = material,
                        .phase = phase,
                    });
                    const std::uint8_t reservation =
                        static_cast<std::uint8_t>(phase + 1);
                    reserveCell(source, reservation);
                    reserveCell(destination, reservation);
                    reserveCell(lowerSurface, reservation);
                    ++transfers;
                }
            }
            }
        }
    }
}

void World::applyLiquidEqualizationPhase(
    const ActiveBounds& bounds, int phase) {
    const auto equalizationBegin =
        std::chrono::steady_clock::now();
    const auto isOpen = [](Material material) {
        return material == Material::air || material == Material::fire ||
               material == Material::smoke || material == Material::steam;
    };
    for (const LiquidEqualizationMove& move :
         liquidEqualizationMoves_) {
        if (static_cast<int>(move.phase) != phase) {
            continue;
        }
        const int sourceX = static_cast<int>(
            move.source % static_cast<std::size_t>(width));
        const int sourceY = static_cast<int>(
            move.source / static_cast<std::size_t>(width));
        const int destinationX = static_cast<int>(
            move.destination % static_cast<std::size_t>(width));
        const int destinationY = static_cast<int>(
            move.destination / static_cast<std::size_t>(width));
        const bool valid =
            sourceX >= bounds.minX && sourceX < bounds.maxX &&
            sourceY >= bounds.minY && sourceY < bounds.maxY &&
            destinationX >= bounds.minX &&
            destinationX < bounds.maxX &&
            destinationY >= bounds.minY &&
            destinationY < bounds.maxY &&
            cells_[move.source] == move.material &&
            cells_[move.support] == move.material &&
            isOpen(cells_[move.destination]) &&
            isOpen(cell(sourceX, sourceY - 1)) &&
            cell(destinationX, destinationY + 1) ==
                move.material;
        if (valid) {
            swapCells(sourceX, sourceY,
                      destinationX, destinationY);
            liquidAmount_[move.destination] =
                maximumLiquidMass;
            const int direction =
                destinationX == sourceX
                    ? 0
                    : (destinationX < sourceX ? -1 : 1);
            liquidFlowX_[move.destination] =
                static_cast<std::int8_t>(
                    direction *
                    (move.material == Material::water ? 88 : 42));
            liquidFlowY_[move.destination] = 0;
            liquidHeadDepth_[move.destination] = 1;
            liquidFoam_[move.destination] = 0;
            moved_[move.source] = 1;
            moved_[move.destination] = 1;
            for (const std::array<int, 2>& center :
                 std::array{
                     std::array{sourceX, sourceY},
                     std::array{destinationX, destinationY}}) {
                for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                        const int candidateX =
                            center[0] + offsetX;
                        const int candidateY =
                            center[1] + offsetY;
                        if (candidateX >= bounds.minX &&
                            candidateX < bounds.maxX &&
                            candidateY >= bounds.minY &&
                            candidateY < bounds.maxY) {
                            const std::size_t candidate =
                                indexOf(candidateX, candidateY);
                            if (liquidFrontierStamp_[candidate] !=
                                liquidFrontierGeneration_) {
                                liquidFrontierStamp_[candidate] =
                                    liquidFrontierGeneration_;
                                liquidWorklist_.push_back(candidate);
                            }
                        }
                    }
                }
            }
        }
        liquidEqualizationReservation_[move.source] = 0;
        liquidEqualizationReservation_[move.destination] = 0;
        liquidEqualizationReservation_[move.support] = 0;
    }
    currentLiquidEqualizationMs_ += elapsedMilliseconds(
        equalizationBegin, std::chrono::steady_clock::now());
}

void World::updateHeat() {
    const ActiveBounds bounds = activeBounds();
    constexpr int chunkColumns =
        (width + chunkSize - 1) / chunkSize;
    const int firstChunkX = bounds.minX / chunkSize;
    const int finalChunkX =
        (bounds.maxX + chunkSize - 1) / chunkSize;
    const int firstChunkY = bounds.minY / chunkSize;
    const int finalChunkY =
        (bounds.maxY + chunkSize - 1) / chunkSize;

    struct ThermalChunkJob {
        int chunkX = 0;
        int chunkY = 0;
        std::vector<std::size_t> cells;
    };
    std::array<std::vector<ThermalChunkJob>, 4> phaseJobs;
    constexpr std::size_t thermalSystem =
        std::countr_zero(
            static_cast<unsigned int>(thermalActivity));
    std::uint32_t thermalChunkCount = 0;
    for (int chunkY = firstChunkY;
         chunkY < finalChunkY; ++chunkY) {
        for (int chunkX = firstChunkX;
             chunkX < finalChunkX; ++chunkX) {
            if (!materialChunkActive(
                    chunkX * chunkSize,
                    chunkY * chunkSize,
                    thermalActivity)) {
                continue;
            }
            const std::size_t chunkIndex =
                static_cast<std::size_t>(
                    chunkY * chunkColumns + chunkX);
            const std::size_t activeCells =
                static_cast<std::size_t>(
                    std::popcount(
                        materialChunkActivity_[chunkIndex]
                            .microtiles[thermalSystem])) *
                static_cast<std::size_t>(
                    materialMicrotileSize *
                    materialMicrotileSize);
            const int phase =
                (chunkX & 1) | ((chunkY & 1) << 1);
            auto& job = phaseJobs[
                static_cast<std::size_t>(phase)].emplace_back();
            job.chunkX = chunkX;
            job.chunkY = chunkY;
            job.cells.reserve(activeCells);
            ++thermalChunkCount;
        }
    }

    ParallelExecutor& executor = materialExecutor();
    materialSimulationTimings_.materialWorkerThreads =
        executor.workerCount();
    materialSimulationTimings_.parallelThermalChunks =
        thermalChunkCount;
    const auto& currentHeat =
        std::as_const(heat_);
    const auto& currentCells =
        std::as_const(cells_);
    constexpr std::array<Vec2, 4> heatOffsets{{
        {-1.0F, 0.0F}, {1.0F, 0.0F},
        {0.0F, -1.0F}, {0.0F, 1.0F},
    }};

    thermalWorklist_.clear();
    for (auto& jobs : phaseJobs) {
        executor.run(
            jobs.size(),
            [&](std::size_t jobIndex) {
                ThermalChunkJob& job = jobs[jobIndex];
                const int chunkOriginX =
                    job.chunkX * chunkSize;
                const int chunkOriginY =
                    job.chunkY * chunkSize;
                const int beginY =
                    std::max(bounds.minY, chunkOriginY);
                const int endY = std::min(
                    bounds.maxY,
                    chunkOriginY + chunkSize);
                for (int y = beginY; y < endY; ++y) {
                    for (int microtileX = 0;
                         microtileX <
                             materialMicrotilesPerAxis;
                         ++microtileX) {
                        const int tileOriginX =
                            chunkOriginX +
                            microtileX *
                                materialMicrotileSize;
                        const int beginX = std::max(
                            bounds.minX, tileOriginX);
                        const int endX = std::min(
                            bounds.maxX,
                            tileOriginX +
                                materialMicrotileSize);
                        if (beginX >= endX ||
                            !materialMicrotileActive(
                                beginX, y,
                                thermalActivity)) {
                            continue;
                        }
                        for (int x = beginX;
                             x < endX; ++x) {
                            const std::size_t index =
                                indexOf(x, y);
                            job.cells.push_back(index);
                            float neighborHeat = 0.0F;
                            int neighborCount = 0;
                            for (Vec2 offset : heatOffsets) {
                                const int neighborX =
                                    x + static_cast<int>(
                                            offset.x);
                                const int neighborY =
                                    y + static_cast<int>(
                                            offset.y);
                                if (neighborX >= 0 &&
                                    neighborX < width &&
                                    neighborY >= 0 &&
                                    neighborY < height) {
                                    neighborHeat +=
                                        currentHeat[indexOf(
                                            neighborX,
                                            neighborY)];
                                    ++neighborCount;
                                }
                            }
                            const float average =
                                neighborCount > 0
                                    ? neighborHeat /
                                          static_cast<float>(
                                              neighborCount)
                                    : 0.0F;
                            float value =
                                (currentHeat[index] * 0.87F +
                                 average * 0.13F) *
                                0.992F;
                            if (currentCells[index] ==
                                Material::water) {
                                value *= 0.72F;
                            }
                            nextHeat_[index] =
                                std::clamp(
                                    value, 0.0F, 1.0F);
                        }
                    }
                }
            });
    }
    for (const auto& jobs : phaseJobs) {
        for (const ThermalChunkJob& job : jobs) {
            thermalWorklist_.insert(
                thermalWorklist_.end(),
                job.cells.begin(), job.cells.end());
        }
    }
    std::sort(thermalWorklist_.begin(),
              thermalWorklist_.end());
    for (std::size_t index : thermalWorklist_) {
        if (nextHeat_[index] > 0.015F) {
            const int x = static_cast<int>(
                index % static_cast<std::size_t>(width));
            const int y = static_cast<int>(
                index / static_cast<std::size_t>(width));
            markMaterialActive(
                x, y, thermalActivity);
        }
        heat_[index] = nextHeat_[index];
    }

    std::uniform_int_distribution<int> percent(0, 99);
    constexpr std::array<Vec2, 4> combustionNeighbors{{
        {0.0F, -1.0F}, {-1.0F, 0.0F},
        {1.0F, 0.0F}, {0.0F, 1.0F},
    }};
    const auto isOpenGas = [](Material material) {
        return material == Material::air || material == Material::fire ||
               material == Material::smoke || material == Material::steam;
    };
    const auto emitFlame = [&](int x, int y) {
        for (Vec2 offset : combustionNeighbors) {
            const int targetX = x + static_cast<int>(offset.x);
            const int targetY = y + static_cast<int>(offset.y);
            const Material target = cell(targetX, targetY);
            if (target == Material::air || target == Material::smoke) {
                setCell(targetX, targetY, Material::fire);
                return;
            }
        }
    };

    const auto updateCombustion =
        [&](int x, int y, std::size_t index) {
            const Material material = cells_[index];
            if (material == Material::water) {
                if (heat_[index] > 0.58F) {
                    setCell(x, y, Material::steam);
                }
                return;
            }
            // Air, terrain, gas, and nonflammable solids do not participate
            // in combustion reactions. Avoid four material-neighbor reads for
            // the overwhelming majority of active cells.
            if (material != Material::wood &&
                material != Material::oil) {
                return;
            }
            bool touchesWater = false;
            bool exposedToAir = false;
            for (Vec2 offset : combustionNeighbors) {
                const Material neighbor =
                    cell(x + static_cast<int>(offset.x),
                         y + static_cast<int>(offset.y));
                touchesWater = touchesWater || neighbor == Material::water;
                exposedToAir = exposedToAir || isOpenGas(neighbor);
            }

            if (material == Material::wood) {
                if (touchesWater) {
                    heat_[index] *= 0.28F;
                }
                if (heat_[index] <= 0.34F) {
                    return;
                }

                // Wood remains a load-bearing solid while it chars. Heat and
                // flame spread from its exposed faces; only after its fuel has
                // been consumed does the cell collapse into smoke.
                heat_[index] = std::max(heat_[index], 0.78F);
                burnProgress_[index] = std::min(
                    1.0F, burnProgress_[index] +
                              0.006F + heat_[index] * 0.006F);
                if (exposedToAir && percent(random_) < 24) {
                    emitFlame(x, y);
                }
                if (burnProgress_[index] >= 1.0F) {
                    setCell(x, y, Material::smoke);
                }
            } else if (material == Material::oil) {
                if (touchesWater) {
                    heat_[index] *= 0.22F;
                }
                if (!exposedToAir || heat_[index] <= 0.40F) {
                    return;
                }

                // Full material cells retain their visual volume while fuel
                // burns. Consumption is tracked separately so oil never
                // becomes a thin fractional stripe.
                heat_[index] = std::max(heat_[index], 0.86F);
                liquidAmount_[index] = maximumLiquidMass;
                burnProgress_[index] = std::min(
                    1.0F, burnProgress_[index] +
                              0.010F + heat_[index] * 0.006F);
                if (burnProgress_[index] >= 1.0F) {
                    setCell(x, y, Material::air);
                } else if (percent(random_) < 42) {
                    emitFlame(x, y);
                }
            }
        };
    for (std::size_t index : thermalWorklist_) {
        const int x = static_cast<int>(
            index % static_cast<std::size_t>(width));
        const int y = static_cast<int>(
            index / static_cast<std::size_t>(width));
        updateCombustion(x, y, index);
    }
    ageMaterialChunks();
}

World::ActiveBounds World::activeBounds() const {
    const Vec2 topLeft = cameraTopLeft();
    const int margin = chunkSize;
    const int rawMinX = static_cast<int>(std::floor(topLeft.x)) - margin;
    const int rawMaxX =
        static_cast<int>(std::ceil(topLeft.x + static_cast<float>(viewWidth))) +
        margin;
    const int rawMinY = static_cast<int>(std::floor(topLeft.y)) - margin;
    const int rawMaxY =
        static_cast<int>(std::ceil(topLeft.y + static_cast<float>(viewHeight))) +
        margin;

    const int minX = std::max(0, (rawMinX / chunkSize) * chunkSize);
    const int minY = std::max(0, (rawMinY / chunkSize) * chunkSize);
    const int maxX = std::min(
        width, ((std::max(rawMaxX, 0) + chunkSize - 1) / chunkSize) * chunkSize);
    const int maxY = std::min(
        height, ((std::max(rawMaxY, 0) + chunkSize - 1) / chunkSize) * chunkSize);
    return {minX, maxX, minY, maxY};
}

void World::releaseEmptySimulationPages() {
    const ActiveBounds bounds = activeBounds();
    const int firstCellX =
        std::max(0, bounds.minX - chunkSize);
    const int firstCellY =
        std::max(0, bounds.minY - chunkSize);
    const int lastCellX =
        std::min(width, bounds.maxX + chunkSize);
    const int lastCellY =
        std::min(height, bounds.maxY + chunkSize);
    const auto release = [&](auto& grid) {
        grid.releaseDefaultPagesOutside(
            firstCellX, firstCellY, lastCellX, lastCellY);
    };
    release(liquidAmount_);
    release(liquidFlowX_);
    release(liquidFlowY_);
    release(liquidHeadDepth_);
    release(liquidFoam_);
    release(liquidEqualizationReservation_);
    release(heat_);
    release(nextHeat_);
    release(burnProgress_);
    release(gasLifetime_);
    release(gasDrift_);
    release(moved_);
    release(liquidFrontierStamp_);
    release(liquidComponentStamp_);
}

void World::updateCamera(float dt) {
    const Vec2 target = player_.position;
    const float smoothing = 1.0F - std::exp(-7.5F * dt);
    cameraCenter_ += (target - cameraCenter_) * smoothing;
    cameraCenter_.x = std::clamp(
        cameraCenter_.x, static_cast<float>(viewWidth) * 0.5F,
        static_cast<float>(width) - static_cast<float>(viewWidth) * 0.5F);
    cameraCenter_.y = std::clamp(
        cameraCenter_.y, static_cast<float>(viewHeight) * 0.5F,
        static_cast<float>(height) - static_cast<float>(viewHeight) * 0.5F);

    explosionFlash_ = std::max(0.0F, explosionFlash_ - dt * 3.8F);
    shakeStrength_ = std::max(0.0F, shakeStrength_ - dt * 13.0F);
    if (shakeStrength_ > 0.01F) {
        std::uniform_real_distribution<float> shake(-shakeStrength_,
                                                    shakeStrength_);
        cameraShake_ = {shake(random_), shake(random_)};
    } else {
        cameraShake_ = {};
    }
}

void World::updateParticles(float dt) {
    if (gpuParticlesEnabled_) {
        return;
    }
    for (Particle& particle : particles_) {
        particle.lifetime -= dt;
        particle.velocity.y += gravity * 0.16F * dt;
        particle.velocity *= std::max(0.0F, 1.0F - dt * 1.8F);
        particle.position += particle.velocity * dt;
    }
    std::erase_if(particles_, [](const Particle& particle) {
        return particle.lifetime <= 0.0F;
    });
}

void World::setGpuParticlesEnabled(bool enabled) {
    if (enabled == gpuParticlesEnabled_) {
        return;
    }
    gpuParticlesEnabled_ = enabled;
    if (enabled) {
        particleSpawns_.insert(particleSpawns_.end(), particles_.begin(),
                               particles_.end());
        particles_.clear();
    }
}

std::vector<Particle> World::takeParticleSpawns() {
    std::vector<Particle> result;
    result.swap(particleSpawns_);
    return result;
}

void World::setGpuMaterialSimulationEnabled(bool enabled) {
    gpuMaterialSimulationEnabled_ = enabled;
    pendingGpuMaterialSteps_ = 0;
}

void World::emitParticle(Particle particle) {
    if (gpuParticlesEnabled_) {
        particleSpawns_.push_back(particle);
    } else {
        particles_.push_back(particle);
    }
}

void World::spawnExplosionParticles(Vec2 center, float radius) {
    std::uniform_real_distribution<float> angle(0.0F, 6.28318F);
    std::uniform_real_distribution<float> speed(radius * 2.2F, radius * 6.2F);
    std::uniform_real_distribution<float> lifetime(0.24F, 0.72F);
    std::uniform_real_distribution<float> size(scaled(0.45F),
                                                scaled(1.45F));
    particles_.reserve(particles_.size() + 72);
    for (int index = 0; index < 72; ++index) {
        const float direction = angle(random_);
        const float velocity = speed(random_);
        const float duration = lifetime(random_);
        const bool ember = index % 3 != 0;
        emitParticle({
            center,
            {std::cos(direction) * velocity, std::sin(direction) * velocity},
            ember ? std::array<float, 3>{1.0F, 0.32F, 0.04F}
                  : std::array<float, 3>{0.42F, 0.40F, 0.38F},
            duration,
            duration,
            size(random_),
        });
    }
}

void World::spawnSplashParticles(Vec2 center, Material material,
                                 float intensity, Vec2 inheritedVelocity) {
    if (!isLiquid(material) || intensity <= 0.0F) {
        return;
    }
    const int count = 4 + static_cast<int>(intensity * 14.0F);
    std::uniform_real_distribution<float> horizontal(-1.0F, 1.0F);
    std::uniform_real_distribution<float> lifetime(0.22F, 0.52F);
    std::uniform_real_distribution<float> size(scaled(0.28F),
                                                scaled(0.72F));
    const std::array<float, 3> color =
        material == Material::water
            ? std::array<float, 3>{0.10F, 0.48F, 0.95F}
            : std::array<float, 3>{0.24F, 0.13F, 0.29F};
    particles_.reserve(particles_.size() + static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        const float xDirection = horizontal(random_);
        const float duration = lifetime(random_);
        emitParticle({
            center,
            inheritedVelocity * 0.18F +
                Vec2{xDirection *
                         (scaled(8.0F) + intensity * scaled(25.0F)),
                     -(scaled(8.0F) + intensity * scaled(28.0F)) *
                         (0.55F + std::abs(xDirection) * 0.45F)},
            color,
            duration,
            duration,
            size(random_),
        });
    }
}

void World::displaceLiquid(Vec2 center, Vec2 halfSize, Vec2 motion,
                           float strength) {
    if (strength <= 0.0F) {
        return;
    }
    const int left = std::max(0, static_cast<int>(
        std::floor(center.x - halfSize.x)));
    const int right = std::min(width - 1, static_cast<int>(
        std::floor(center.x + halfSize.x)));
    const int top = std::max(0, static_cast<int>(
        std::floor(center.y - halfSize.y)));
    const int bottom = std::min(height - 1, static_cast<int>(
        std::floor(center.y + halfSize.y)));
    std::vector<std::pair<int, int>> sources;
    sources.reserve(static_cast<std::size_t>(
        (right - left + 1) * (bottom - top + 1)));
    for (int y = top; y <= bottom; ++y) {
        for (int x = left; x <= right; ++x) {
            const std::size_t source = indexOf(x, y);
            if (isLiquid(cells_[source]) &&
                liquidAmount_[source] != 0) {
                sources.emplace_back(x, y);
            }
        }
    }
    if (sources.empty()) {
        return;
    }

    // Displacement is an impulse, not a second fluid solver. The old path
    // evacuated every liquid cell inside the body on every call and placed
    // successive scan rows on opposite sides. During vertical movement that
    // manufactured the enormous comb/checkerboard plumes seen on screen.
    std::shuffle(sources.begin(), sources.end(), random_);
    const float motionLength = length(motion);
    const int maximumMoves = std::clamp(
        static_cast<int>(std::ceil(
            strength * (1.0F + motionLength) * 1.5F)),
        1, scaledCell(4));
    const Vec2 motionDirection =
        motionLength > 0.01F ? motion * (1.0F / motionLength)
                             : Vec2{0.0F, -1.0F};
    const bool verticalMotion =
        std::abs(motionDirection.y) >= std::abs(motionDirection.x);
    constexpr std::array<int, 9> fanOffsets{
        0, -1, 1, -2, 2, -3, 3, -4, 4,
    };
    int movedCells = 0;
    for (const auto& [x, y] : sources) {
        if (movedCells >= maximumMoves) {
            break;
        }

        int baseX = x;
        int baseY = y;
        Vec2 ejectionDirection{};
        bool fanVertically = false;
        if (verticalMotion) {
            const int side =
                static_cast<float>(x) + 0.5F < center.x ? -1 : 1;
            baseX = side < 0 ? left - 1 : right + 1;
            ejectionDirection = normalized(
                Vec2{static_cast<float>(side),
                     motionDirection.y * 0.22F});
            fanVertically = true;
        } else {
            const int side =
                static_cast<float>(y) + 0.5F < center.y ? -1 : 1;
            baseY = side < 0 ? top - 1 : bottom + 1;
            ejectionDirection = normalized(
                Vec2{motionDirection.x * 0.22F,
                     static_cast<float>(side)});
        }

        for (int fanOffset : fanOffsets) {
            const int targetX =
                baseX + (fanVertically ? 0 : fanOffset);
            const int targetY =
                baseY + (fanVertically ? fanOffset : 0);
            if (targetX < 0 || targetX >= width ||
                targetY < 0 || targetY >= height) {
                continue;
            }
            const Material targetMaterial = cell(targetX, targetY);
            if (targetMaterial != Material::air &&
                targetMaterial != Material::smoke &&
                targetMaterial != Material::steam &&
                targetMaterial != Material::fire) {
                continue;
            }

            const Material material = cell(x, y);
            swapCells(x, y, targetX, targetY);
            const std::size_t target = indexOf(targetX, targetY);
            liquidAmount_[target] = maximumLiquidMass;
            heat_[target] *=
                material == Material::water ? 0.35F : 0.8F;
            liquidFlowX_[target] = static_cast<std::int8_t>(
                std::clamp(
                    static_cast<int>(ejectionDirection.x * 96.0F),
                    -127, 127));
            liquidFlowY_[target] = static_cast<std::int8_t>(
                std::clamp(
                    static_cast<int>(ejectionDirection.y * 96.0F),
                    -127, 127));
            ++movedCells;
            break;
        }
    }
}

void World::fireBullet(Vec2 direction) {
    bullets_.push_back({
        player_.position + direction * scaled(5.2F),
        direction * scaled(250.0F),
        1.3F,
        1.0F,
    });
}

void World::throwGrenade(Vec2 direction) {
    grenades_.push_back({
        player_.position + direction * scaled(6.0F),
        direction * scaled(82.0F) + Vec2{0.0F, scaled(-10.0F)},
        1.65F,
        1.0F,
    });
}

} // namespace gunpowder
