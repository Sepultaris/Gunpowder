#include "game/parallel_executor.hpp"
#include "game/world.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <vector>

namespace {

std::uint64_t massOf(const gunpowder::World& world,
                     gunpowder::Material material) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < world.materials().size(); ++index) {
        if (world.materials()[index] == material) {
            result += world.liquidAmounts()[index];
        }
    }
    return result;
}

std::size_t countOf(const gunpowder::World& world,
                    gunpowder::Material material) {
    return static_cast<std::size_t>(std::count(
        world.materials().begin(), world.materials().end(), material));
}

std::size_t unsupportedHorizontalLiquidCells(
    const gunpowder::World& world) {
    const auto isOpen = [](gunpowder::Material material) {
        return material == gunpowder::Material::air ||
               material == gunpowder::Material::fire ||
               material == gunpowder::Material::smoke ||
               material == gunpowder::Material::steam;
    };
    std::size_t result = 0;
    for (int y = 1; y < gunpowder::World::height - 1; ++y) {
        for (int x = 1; x < gunpowder::World::width - 1; ++x) {
            const gunpowder::Material material = world.cell(x, y);
            if (material != gunpowder::Material::water &&
                material != gunpowder::Material::oil) {
                continue;
            }
            const bool horizontalNeighbor =
                world.cell(x - 1, y) == material ||
                world.cell(x + 1, y) == material;
            if (horizontalNeighbor &&
                isOpen(world.cell(x, y - 1)) &&
                isOpen(world.cell(x, y + 1))) {
                ++result;
            }
        }
    }
    return result;
}

std::size_t singleCellVerticalLiquidGaps(
    const gunpowder::World& world) {
    std::size_t result = 0;
    for (int y = 1; y < gunpowder::World::height - 1; ++y) {
        for (int x = 1; x < gunpowder::World::width - 1; ++x) {
            const gunpowder::Material above = world.cell(x, y - 1);
            if (above != gunpowder::Material::water &&
                above != gunpowder::Material::oil) {
                continue;
            }
            if (world.cell(x, y) == gunpowder::Material::air &&
                world.cell(x, y + 1) == above) {
                ++result;
            }
        }
    }
    return result;
}

float averageLiquidSurface(const gunpowder::World& world,
                           gunpowder::Material material,
                           int minX, int maxX,
                           int minY, int maxY) {
    float total = 0.0F;
    int columns = 0;
    for (int x = minX; x <= maxX; ++x) {
        for (int y = minY; y <= maxY; ++y) {
            if (world.cell(x, y) == material) {
                total += static_cast<float>(y);
                ++columns;
                break;
            }
        }
    }
    return columns > 0 ? total / static_cast<float>(columns)
                       : static_cast<float>(maxY + 1);
}

} // namespace

int main() {
    {
        gunpowder::ParallelExecutor executor(4);
        std::array<std::atomic<std::uint32_t>, 257> visits{};
        executor.run(visits.size(), [&](std::size_t index) {
            visits[index].fetch_add(1, std::memory_order_relaxed);
        });
        for (const auto& visitCount : visits) {
            if (visitCount.load(std::memory_order_relaxed) != 1) {
                std::cerr
                    << "Parallel executor skipped or repeated a job\n";
                return 1;
            }
        }
    }

    gunpowder::ChunkGrid<std::uint8_t> spatialChunks(
        130, 130, 0);
    spatialChunks.set(63, 63, 11);
    spatialChunks.set(64, 64, 22);
    spatialChunks.set(129, 129, 33);
    spatialChunks.set(128, 0, 0);
    if (!spatialChunks.hasChunk(0, 0) ||
        !spatialChunks.hasChunk(1, 1) ||
        !spatialChunks.hasChunk(2, 2) ||
        spatialChunks.hasChunk(2, 0) ||
        spatialChunks.allocatedChunkCount() != 3) {
        std::cerr << "Spatial chunk allocation crossed a chunk boundary\n";
        return 1;
    }
    if (spatialChunks.get(63, 63) != 11 ||
        spatialChunks.get(64, 64) != 22 ||
        spatialChunks.get(129, 129) != 33 ||
        spatialChunks.get(0, 129) != 0) {
        std::cerr << "Spatial chunk address mapping is incorrect\n";
        return 1;
    }

    std::array<bool, 9> visitedChunks{};
    spatialChunks.forEachAllocatedChunk(
        [&](int chunkX, int chunkY, const auto&) {
            visitedChunks[static_cast<std::size_t>(
                chunkY * spatialChunks.chunksWide() + chunkX)] = true;
        });
    if (!visitedChunks[0] || !visitedChunks[4] ||
        !visitedChunks[8]) {
        std::cerr << "Allocated chunk traversal lost spatial coordinates\n";
        return 1;
    }

    spatialChunks.set(129, 129, 0);
    spatialChunks.releaseDefaultChunksOutside(0, 0, 64, 64);
    if (spatialChunks.hasChunk(2, 2) ||
        !spatialChunks.hasChunk(0, 0) ||
        spatialChunks.get(64, 64) != 22) {
        std::cerr << "Default chunk reclamation removed persistent state\n";
        return 1;
    }

    const auto copiedSpatialChunks = spatialChunks;
    spatialChunks.reset(7);
    if (copiedSpatialChunks.get(64, 64) != 22 ||
        spatialChunks.get(64, 64) != 7 ||
        spatialChunks.allocatedChunkCount() != 0) {
        std::cerr << "Spatial chunk copy/reset was not isolated\n";
        return 1;
    }

    gunpowder::World activityWorld;
    constexpr int activityX = 100;
    constexpr int activityY = 10;
    activityWorld.clearMaterialActivityForTest();
    activityWorld.setCellForTest(
        activityX, activityY, gunpowder::Material::water);
    const auto waterActivity =
        activityWorld.materialActivityForTest(
            activityX, activityY);
    if (waterActivity[0] || !waterActivity[1] ||
        waterActivity[2] || !waterActivity[3]) {
        std::cerr << "Water woke unrelated material systems\n";
        return 1;
    }
    const auto waterMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX, activityY);
    const auto distantMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX + 24, activityY);
    if (waterMicrotile[0] || !waterMicrotile[1] ||
        waterMicrotile[2] || !waterMicrotile[3] ||
        distantMicrotile[0] || distantMicrotile[1] ||
        distantMicrotile[2] || distantMicrotile[3]) {
        std::cerr << "Material wake escaped its microtile halo\n";
        return 1;
    }

    constexpr int microtileBoundaryX = 64;
    activityWorld.clearMaterialActivityForTest();
    activityWorld.setCellForTest(
        microtileBoundaryX, activityY,
        gunpowder::Material::water);
    if (!activityWorld.materialMicrotileActivityForTest(
             microtileBoundaryX - 8, activityY)[1] ||
        !activityWorld.materialMicrotileActivityForTest(
             microtileBoundaryX, activityY)[1] ||
        !activityWorld.materialMicrotileActivityForTest(
             microtileBoundaryX + 8, activityY)[1] ||
        activityWorld.materialMicrotileActivityForTest(
            microtileBoundaryX + 24, activityY)[1]) {
        std::cerr << "Microtile halo did not cross a chunk boundary\n";
        return 1;
    }

    activityWorld.setCellForTest(
        activityX, activityY, gunpowder::Material::air);
    activityWorld.clearMaterialActivityForTest();
    activityWorld.setCellForTest(
        activityX, activityY, gunpowder::Material::smoke);
    const auto smokeActivity =
        activityWorld.materialActivityForTest(
            activityX, activityY);
    if (smokeActivity[0] || smokeActivity[1] ||
        !smokeActivity[2] || !smokeActivity[3]) {
        std::cerr << "Smoke woke unrelated material systems\n";
        return 1;
    }
    const auto smokeMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX, activityY);
    const auto distantSmokeMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX + 24, activityY);
    if (smokeMicrotile[0] || smokeMicrotile[1] ||
        !smokeMicrotile[2] || !smokeMicrotile[3] ||
        distantSmokeMicrotile[2] ||
        distantSmokeMicrotile[3]) {
        std::cerr << "Smoke wake escaped its microtile halo\n";
        return 1;
    }

    activityWorld.setCellForTest(
        activityX, activityY, gunpowder::Material::air);
    activityWorld.clearMaterialActivityForTest();
    activityWorld.setCellForTest(
        activityX, activityY, gunpowder::Material::sand);
    const auto topologyActivity =
        activityWorld.materialActivityForTest(
            activityX, activityY);
    if (!topologyActivity[0] || !topologyActivity[1] ||
        !topologyActivity[2] || !topologyActivity[3]) {
        std::cerr << "A changed solid boundary did not wake its neighbors\n";
        return 1;
    }
    const auto sandMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX, activityY);
    const auto distantSandMicrotile =
        activityWorld.materialMicrotileActivityForTest(
            activityX + 24, activityY);
    if (!sandMicrotile[0] || !sandMicrotile[1] ||
        !sandMicrotile[2] || !sandMicrotile[3] ||
        distantSandMicrotile[0] ||
        distantSandMicrotile[1] ||
        distantSandMicrotile[2] ||
        distantSandMicrotile[3]) {
        std::cerr << "Solid-boundary wake escaped its microtile halo\n";
        return 1;
    }

    gunpowder::World chunkHeadWorld;
    constexpr int headColumnX = 100;
    constexpr int headBottomY = 71;
    chunkHeadWorld.setPlayerForTest(
        {static_cast<float>(headColumnX), 48.0F});
    for (int y = 0; y <= headBottomY; ++y) {
        chunkHeadWorld.setCellForTest(
            headColumnX - 1, y, gunpowder::Material::stone);
        chunkHeadWorld.setCellForTest(
            headColumnX + 1, y, gunpowder::Material::stone);
    }
    chunkHeadWorld.setCellForTest(
        headColumnX, headBottomY, gunpowder::Material::stone);
    for (int y = 0; y < headBottomY; ++y) {
        chunkHeadWorld.setCellForTest(
            headColumnX, y, gunpowder::Material::water);
    }
    chunkHeadWorld.setCellForTest(
        headColumnX, 0, gunpowder::Material::air);
    chunkHeadWorld.setCellForTest(
        headColumnX, 64, gunpowder::Material::air);
    chunkHeadWorld.clearMaterialActivityForTest();
    chunkHeadWorld.setCellForTest(
        headColumnX, 0, gunpowder::Material::water);
    chunkHeadWorld.setCellForTest(
        headColumnX, 64, gunpowder::Material::water);
    gunpowder::InputState chunkHeadInput;
    chunkHeadWorld.update(1.0F / 30.0F, chunkHeadInput);
    if (chunkHeadWorld.liquidHeadDepthForTest(
            headColumnX, 63) != 64 ||
        chunkHeadWorld.liquidHeadDepthForTest(
            headColumnX, 64) != 65 ||
        chunkHeadWorld.liquidHeadDepthForTest(
            headColumnX, 70) != 71) {
        std::cerr << "Liquid head depth broke across a chunk boundary\n";
        return 1;
    }
    const auto& chunkHeadTimings =
        chunkHeadWorld.materialSimulationTimings();
    const std::uint32_t visibleCellCount =
        static_cast<std::uint32_t>(
            gunpowder::World::viewWidth *
            gunpowder::World::viewHeight);
    if (chunkHeadTimings.liquidPreparationCellVisits == 0 ||
        chunkHeadTimings.liquidPreparationCellVisits >=
            visibleCellCount ||
        chunkHeadTimings.liquidHeadSummaryHits == 0 ||
        chunkHeadTimings.liquidEqualizationSeedVisits == 0 ||
        chunkHeadTimings.liquidEqualizationSeedVisits >=
            visibleCellCount) {
        std::cerr
            << "Liquid preparation fell back to a camera-wide scan: prep "
            << chunkHeadTimings.liquidPreparationCellVisits
            << ", seeds "
            << chunkHeadTimings.liquidEqualizationSeedVisits
            << ", visible " << visibleCellCount << '\n';
        return 1;
    }

    gunpowder::SparseGrid<std::uint8_t> streamedState(
        4096, 2048, 0);
    const std::size_t distantState =
        static_cast<std::size_t>(1700 * 4096 + 3500);
    streamedState.set(distantState, 73);
    streamedState.releaseDefaultPagesOutside(0, 0, 2, 2);
    if (streamedState[distantState] != 73 ||
        streamedState[0] != 0) {
        std::cerr << "Sparse world page lost distant persistent state\n";
        return 1;
    }
    const auto copiedStreamedState = streamedState;
    streamedState.reset(0);
    if (copiedStreamedState[distantState] != 73 ||
        streamedState[distantState] != 0) {
        std::cerr << "Sparse world page copy/reset was not isolated\n";
        return 1;
    }

    gunpowder::World skyWorld;
    constexpr int skyTestX = gunpowder::World::width / 2;
    constexpr int skyTestRoofY = 20;
    for (int y = 0; y <= skyTestRoofY; ++y) {
        skyWorld.setCellForTest(
            skyTestX, y, gunpowder::Material::air);
    }
    skyWorld.setCellForTest(
        skyTestX, skyTestRoofY, gunpowder::Material::stone);
    gunpowder::InputState skyInput;
    skyWorld.update(1.0F / 120.0F, skyInput);
    if (skyWorld.skyOccluderY(skyTestX) != skyTestRoofY) {
        std::cerr << "Sky visibility cache missed an inserted roof\n";
        return 1;
    }
    skyWorld.setCellForTest(
        skyTestX, skyTestRoofY, gunpowder::Material::air);
    skyWorld.update(1.0F / 120.0F, skyInput);
    if (skyWorld.skyOccluderY(skyTestX) <= skyTestRoofY) {
        std::cerr << "Sky visibility cache retained a destroyed roof\n";
        return 1;
    }
    skyWorld.setCellForTest(
        skyTestX, skyTestRoofY, gunpowder::Material::metal);
    skyWorld.update(1.0F / 120.0F, skyInput);
    if (skyWorld.skyOccluderY(skyTestX) != skyTestRoofY) {
        std::cerr << "Metal was not treated as an opaque solid\n";
        return 1;
    }
    skyWorld.setCellForTest(
        skyTestX, skyTestRoofY, gunpowder::Material::air);
    skyWorld.update(1.0F / 120.0F, skyInput);

    constexpr int tombInteriorX = 48;
    constexpr int tombInteriorY = 80;
    if (!skyWorld.hasInteriorBackdrop(
            tombInteriorX, tombInteriorY)) {
        std::cerr << "Generated tomb room lacks an interior backdrop\n";
        return 1;
    }
    for (int y = 0; y <= tombInteriorY; ++y) {
        skyWorld.setCellForTest(
            tombInteriorX, y, gunpowder::Material::air);
    }
    skyWorld.update(1.0F / 120.0F, skyInput);
    if (skyWorld.skyOccluderY(tombInteriorX) <= tombInteriorY) {
        std::cerr << "Destroyed tomb roof did not open to sunlight\n";
        return 1;
    }
    if (!skyWorld.hasInteriorBackdrop(
            tombInteriorX, tombInteriorY)) {
        std::cerr << "Destroyed tomb roof erased the interior backdrop\n";
        return 1;
    }

    gunpowder::World sunShadowWorld;
    constexpr int shadowTestX = 10;
    constexpr int shadowRoofY = 20;
    constexpr int shadowSampleY = 38;
    for (int y = 0; y <= shadowSampleY; ++y) {
        sunShadowWorld.setCellForTest(
            shadowTestX, y, gunpowder::Material::air);
    }
    sunShadowWorld.setCellForTest(
        shadowTestX, shadowRoofY, gunpowder::Material::stone);
    constexpr float horizonSamplesPerCell = 4.0F;
    std::vector<float> sunHorizonDepths;
    std::vector<std::int32_t> sunHorizonBlockers;
    float sunHorizonMinimum = 0.0F;
    gunpowder::Vec2 sunDirection{0.0F, -1.0F};
    sunShadowWorld.buildDirectionalSunHorizon(
        sunDirection, horizonSamplesPerCell,
        sunHorizonDepths, sunHorizonBlockers,
        sunHorizonMinimum);
    const auto blockerAt = [&](int x, int y) {
        const gunpowder::Vec2 direction =
            gunpowder::normalized(sunDirection);
        const float centerX = static_cast<float>(x) + 0.5F;
        const float centerY = static_cast<float>(y) + 0.5F;
        const float perpendicular =
            -direction.y * centerX + direction.x * centerY;
        const int sample = static_cast<int>(std::floor(
            (perpendicular - sunHorizonMinimum) *
            horizonSamplesPerCell));
        const std::int32_t cellIndex =
            y * gunpowder::World::width + x;
        if (sample < 0 ||
            sample >= static_cast<int>(sunHorizonBlockers.size())) {
            return std::int32_t{-1};
        }
        const std::int32_t blocker =
            sunHorizonBlockers[static_cast<std::size_t>(sample)];
        if (blocker < 0 || blocker == cellIndex) {
            return std::int32_t{-1};
        }
        const float receiverDepth =
            direction.x * centerX + direction.y * centerY +
            ([&] {
                const gunpowder::Material material =
                    sunShadowWorld.cell(x, y);
                return material == gunpowder::Material::dirt ||
                       material == gunpowder::Material::sand ||
                       material == gunpowder::Material::rock ||
                       material == gunpowder::Material::wood ||
                       material == gunpowder::Material::stone ||
                       material == gunpowder::Material::metal;
            }()
                 ? 0.5F *
                       (std::abs(direction.x) +
                        std::abs(direction.y))
                 : 0.0F);
        return sunHorizonDepths[
                   static_cast<std::size_t>(sample)] >
                       receiverDepth + 0.001F
                   ? blocker
                   : std::int32_t{-1};
    };
    const std::int32_t expectedRoofIndex =
        shadowRoofY * gunpowder::World::width + shadowTestX;
    if (blockerAt(shadowTestX, shadowRoofY) != -1 ||
        blockerAt(shadowTestX, shadowSampleY) != expectedRoofIndex) {
        std::cerr << "Whole-world sun cache did not shadow below a roof\n";
        return 1;
    }
    sunShadowWorld.setCellForTest(
        shadowTestX, shadowRoofY, gunpowder::Material::air);
    sunShadowWorld.buildDirectionalSunHorizon(
        sunDirection, horizonSamplesPerCell,
        sunHorizonDepths, sunHorizonBlockers,
        sunHorizonMinimum);
    if (blockerAt(shadowTestX, shadowSampleY) != -1) {
        std::cerr << "Whole-world sun cache retained a destroyed roof\n";
        return 1;
    }

    for (int y = 0; y <= 40; ++y) {
        for (int x = 0; x <= 50; ++x) {
            sunShadowWorld.setCellForTest(
                x, y, gunpowder::Material::air);
        }
    }
    constexpr int diagonalBlockerX = 20;
    constexpr int diagonalBlockerY = 20;
    constexpr int diagonalShadowX = 14;
    constexpr int diagonalShadowY = 28;
    sunShadowWorld.setCellForTest(
        diagonalBlockerX, diagonalBlockerY,
        gunpowder::Material::stone);
    sunDirection = {0.6F, -0.8F};
    sunShadowWorld.buildDirectionalSunHorizon(
        sunDirection, horizonSamplesPerCell,
        sunHorizonDepths, sunHorizonBlockers,
        sunHorizonMinimum);
    const std::int32_t expectedDiagonalBlocker =
        diagonalBlockerY * gunpowder::World::width +
        diagonalBlockerX;
    if (blockerAt(diagonalShadowX, diagonalShadowY) !=
        expectedDiagonalBlocker) {
        std::cerr << "Sun horizon leaked through a diagonal cell path\n";
        return 1;
    }

    for (int x = 8; x <= 32; ++x) {
        sunShadowWorld.setCellForTest(
            x, 20, gunpowder::Material::stone);
    }
    sunDirection = {0.52F, -0.8541667F};
    sunShadowWorld.buildDirectionalSunHorizon(
        sunDirection, horizonSamplesPerCell,
        sunHorizonDepths, sunHorizonBlockers,
        sunHorizonMinimum);
    for (int x = 8; x <= 32; ++x) {
        if (blockerAt(x, 20) != -1) {
            std::cerr << "Oblique sun made a roof cell shadow its neighbor\n";
            return 1;
        }
    }

    for (int y = 0; y <= 40; ++y) {
        for (int x = 0; x <= 130; ++x) {
            sunShadowWorld.setCellForTest(
                x, y, gunpowder::Material::air);
        }
    }
    for (int x = 0; x <= 120; ++x) {
        sunShadowWorld.setCellForTest(
            x, 20, gunpowder::Material::stone);
    }
    sunDirection = {0.98F, -0.1989975F};
    sunShadowWorld.buildDirectionalSunHorizon(
        sunDirection, horizonSamplesPerCell,
        sunHorizonDepths, sunHorizonBlockers,
        sunHorizonMinimum);
    for (int x = 10; x <= 40; ++x) {
        if (blockerAt(x, 30) == -1) {
            std::cerr << "Shallow sun produced gaps beneath a solid roof\n";
            return 1;
        }
    }

    gunpowder::World world;
    if (countOf(world, gunpowder::Material::wood) < 100 ||
        countOf(world, gunpowder::Material::stone) < 1000) {
        std::cerr << "Starting tomb is missing its structural materials\n";
        return 1;
    }

    gunpowder::World matchingSeedWorld;
    if (world.materials() != matchingSeedWorld.materials()) {
        std::cerr << "World generation is not deterministic for its seed\n";
        return 1;
    }
    const auto firstGeneratedWorld =
        matchingSeedWorld.materials();
    matchingSeedWorld.regenerate();
    if (firstGeneratedWorld == matchingSeedWorld.materials()) {
        std::cerr << "Regeneration did not advance the procedural world\n";
        return 1;
    }

    // Wood may be destroyed by the player's starting weapon, so include it
    // when checking whether the main generated tomb wing can be opened.
    const auto generationPassable = [](gunpowder::Material material) {
        return material == gunpowder::Material::air ||
               material == gunpowder::Material::water ||
               material == gunpowder::Material::oil ||
               material == gunpowder::Material::fire ||
               material == gunpowder::Material::smoke ||
               material == gunpowder::Material::steam ||
               material == gunpowder::Material::wood;
    };
    std::vector<std::uint8_t> generationVisited(
        static_cast<std::size_t>(
            gunpowder::World::width * gunpowder::World::height),
        0);
    std::deque<std::pair<int, int>> generationFrontier;
    const int generationStartX =
        static_cast<int>(std::floor(world.player().position.x));
    const int generationStartY =
        static_cast<int>(std::floor(world.player().position.y));
    generationFrontier.emplace_back(
        generationStartX, generationStartY);
    generationVisited[static_cast<std::size_t>(
        generationStartY * gunpowder::World::width +
        generationStartX)] = 1;
    int furthestGeneratedX = generationStartX;
    constexpr std::array<std::pair<int, int>, 4> generationNeighbors{
        std::pair{-1, 0},
        std::pair{1, 0},
        std::pair{0, -1},
        std::pair{0, 1},
    };
    while (!generationFrontier.empty()) {
        const auto [x, y] = generationFrontier.front();
        generationFrontier.pop_front();
        furthestGeneratedX = std::max(furthestGeneratedX, x);
        for (const auto [offsetX, offsetY] : generationNeighbors) {
            const int neighborX = x + offsetX;
            const int neighborY = y + offsetY;
            if (neighborX < 0 ||
                neighborX >= gunpowder::World::width ||
                neighborY < 0 ||
                neighborY >= gunpowder::World::height ||
                !generationPassable(
                    world.cell(neighborX, neighborY))) {
                continue;
            }
            const std::size_t neighborIndex =
                static_cast<std::size_t>(
                    neighborY * gunpowder::World::width +
                    neighborX);
            if (generationVisited[neighborIndex] != 0) {
                continue;
            }
            generationVisited[neighborIndex] = 1;
            generationFrontier.emplace_back(
                neighborX, neighborY);
        }
    }
    if (furthestGeneratedX < 900) {
        std::cerr << "Generated tomb wing is sealed from the starting room\n";
        return 1;
    }

    const std::uint64_t initialWater =
        massOf(world, gunpowder::Material::water);
    const std::uint64_t initialOil = massOf(world, gunpowder::Material::oil);
    bool observedFlow = false;
    bool observedPartialCell = false;

    gunpowder::InputState input;
    input.aim = {160.0F, 90.0F};
    for (int tick = 0; tick < 600; ++tick) {
        world.update(1.0F / 120.0F, input);
        if (tick % 4 != 0) {
            continue;
        }
        for (std::size_t index = 0; index < world.materials().size(); ++index) {
            if (world.liquidFlowX()[index] != 0 ||
                world.liquidFlowY()[index] != 0) {
                observedFlow = true;
            }
            const std::uint16_t amount = world.liquidAmounts()[index];
            if (amount > 0 && amount < 255) {
                observedPartialCell = true;
            }
        }
    }

    const std::uint64_t finalWater =
        massOf(world, gunpowder::Material::water);
    const std::uint64_t finalOil = massOf(world, gunpowder::Material::oil);
    if (initialWater != finalWater || initialOil != finalOil) {
        std::cerr << "Liquid mass changed without a phase reaction\n";
        return 1;
    }
    if (!observedFlow) {
        std::cerr << "No liquid flow was observed\n";
        return 1;
    }
    if (observedPartialCell) {
        std::cerr << "The cellular liquid solver produced a fractional cell\n";
        return 1;
    }

    input.aim = world.player().position;
    input.paint = true;
    input.paintMaterial = gunpowder::Material::water;
    world.update(1.0F / 120.0F, input);
    input.paint = false;
    input.moveRight = true;
    input.swimUp = true;
    const std::uint64_t waterBeforeDisplacement =
        massOf(world, gunpowder::Material::water);
    bool playerBecameWet = false;
    std::size_t maximumUnsupportedShelfCells = 0;
    std::size_t maximumVerticalLiquidGaps = 0;
    for (int tick = 0; tick < 240; ++tick) {
        world.update(1.0F / 120.0F, input);
        playerBecameWet =
            playerBecameWet || world.player().wetness > 0.05F;
        maximumUnsupportedShelfCells = std::max(
            maximumUnsupportedShelfCells,
            unsupportedHorizontalLiquidCells(world));
        maximumVerticalLiquidGaps = std::max(
            maximumVerticalLiquidGaps,
            singleCellVerticalLiquidGaps(world));
    }
    if (massOf(world, gunpowder::Material::water) !=
        waterBeforeDisplacement) {
        std::cerr << "Entity displacement changed water mass\n";
        return 1;
    }
    if (!playerBecameWet) {
        std::cerr << "Player exposure did not detect painted water\n";
        return 1;
    }
    const std::size_t liquidCellCount =
        countOf(world, gunpowder::Material::water) +
        countOf(world, gunpowder::Material::oil);
    if (maximumUnsupportedShelfCells >
        std::max<std::size_t>(8, liquidCellCount / 8)) {
        std::cerr
            << "Player/liquid interaction produced unsupported horizontal "
               "shelves ("
            << maximumUnsupportedShelfCells << " cells)\n";
        return 1;
    }
    if (maximumVerticalLiquidGaps >
        std::max<std::size_t>(6, liquidCellCount / 12)) {
        std::cerr
            << "Player/liquid interaction split narrow streams into "
               "alternating cells ("
            << maximumVerticalLiquidGaps << " gaps)\n";
        return 1;
    }

    // Two reservoirs with different surface elevations share a submerged
    // connection beneath the divider. Their free surfaces must converge even
    // though no same-height air corridor joins the two sides.
    gunpowder::World equalizationWorld;
    for (int y = 20; y <= 160; ++y) {
        for (int x = 180; x <= 300; ++x) {
            equalizationWorld.setCellForTest(
                x, y, gunpowder::Material::rock);
        }
    }
    for (int y = 22; y <= 158; ++y) {
        for (int x = 182; x <= 298; ++x) {
            equalizationWorld.setCellForTest(
                x, y, gunpowder::Material::air);
        }
    }
    for (int y = 22; y <= 92; ++y) {
        equalizationWorld.setCellForTest(
            240, y, gunpowder::Material::stone);
    }
    for (int y = 50; y <= 158; ++y) {
        for (int x = 182; x <= 239; ++x) {
            equalizationWorld.setCellForTest(
                x, y, gunpowder::Material::water);
        }
    }
    for (int y = 100; y <= 158; ++y) {
        for (int x = 241; x <= 298; ++x) {
            equalizationWorld.setCellForTest(
                x, y, gunpowder::Material::water);
        }
    }
    for (int y = 93; y <= 158; ++y) {
        equalizationWorld.setCellForTest(
            240, y, gunpowder::Material::water);
    }
    const std::uint64_t equalizationMass =
        massOf(equalizationWorld, gunpowder::Material::water);
    gunpowder::InputState equalizationInput;
    for (int tick = 0; tick < 600; ++tick) {
        equalizationWorld.update(
            1.0F / 120.0F, equalizationInput);
    }
    const float leftSurface = averageLiquidSurface(
        equalizationWorld, gunpowder::Material::water,
        182, 239, 22, 158);
    const float rightSurface = averageLiquidSurface(
        equalizationWorld, gunpowder::Material::water,
        241, 298, 22, 158);
    if (std::abs(leftSurface - rightSurface) > 3.0F) {
        std::cerr
            << "Connected water reservoirs did not equalize ("
            << leftSurface << " versus " << rightSurface
            << ")\n";
        return 1;
    }
    if (massOf(equalizationWorld, gunpowder::Material::water) !=
        equalizationMass) {
        std::cerr
            << "Connected-reservoir equalization changed water mass\n";
        return 1;
    }
    std::size_t movingSettledWater = 0;
    for (int y = 22; y <= 158; ++y) {
        for (int x = 182; x <= 298; ++x) {
            const std::size_t index = static_cast<std::size_t>(
                y * gunpowder::World::width + x);
            if (equalizationWorld.cell(x, y) ==
                    gunpowder::Material::water &&
                (equalizationWorld.liquidFlowX()[index] != 0 ||
                 equalizationWorld.liquidFlowY()[index] != 0)) {
                ++movingSettledWater;
            }
        }
    }
    if (movingSettledWater != 0) {
        std::cerr
            << "Equalized water retained residual surface motion ("
            << movingSettledWater << " cells)\n";
        return 1;
    }

    gunpowder::World firstThermalWorld;
    gunpowder::World secondThermalWorld;
    constexpr std::array<std::pair<int, int>, 4> heatSources{{
        {96, 80},
        {160, 80},
        {96, 144},
        {160, 144},
    }};
    for (const auto& [x, y] : heatSources) {
        firstThermalWorld.setCellForTest(
            x, y, gunpowder::Material::fire);
        secondThermalWorld.setCellForTest(
            x, y, gunpowder::Material::fire);
    }
    gunpowder::InputState thermalInput;
    for (int tick = 0; tick < 12; ++tick) {
        firstThermalWorld.update(1.0F / 30.0F, thermalInput);
        secondThermalWorld.update(1.0F / 30.0F, thermalInput);
    }
    for (std::size_t index = 0;
         index < firstThermalWorld.materials().size(); ++index) {
        if (firstThermalWorld.materials()[index] !=
            secondThermalWorld.materials()[index]) {
            std::cerr
                << "Parallel thermal scheduling changed material at "
                << index << '\n';
            return 1;
        }
        if (firstThermalWorld.heat()[index] !=
            secondThermalWorld.heat()[index]) {
            std::cerr
                << "Parallel thermal scheduling changed heat at "
                << index << " ("
                << firstThermalWorld.heat()[index] << " versus "
                << secondThermalWorld.heat()[index] << ")\n";
            return 1;
        }
    }
    const auto& thermalTimings =
        firstThermalWorld.materialSimulationTimings();
    if (thermalTimings.parallelThermalChunks < 2 ||
        thermalTimings.materialWorkerThreads < 1) {
        std::cerr
            << "Thermal work did not reach the parallel chunk scheduler\n";
        return 1;
    }

    gunpowder::World burnWorld;
    gunpowder::InputState burnInput;
    burnInput.paint = true;
    burnInput.paintMaterial = gunpowder::Material::fire;
    burnInput.aim = {66.5F, 105.0F};
    burnWorld.update(1.0F / 120.0F, burnInput);
    burnInput.paint = false;
    const std::size_t woodAfterIgnition =
        countOf(burnWorld, gunpowder::Material::wood);
    for (int tick = 0; tick < 600; ++tick) {
        burnWorld.update(1.0F / 120.0F, burnInput);
    }
    if (countOf(burnWorld, gunpowder::Material::wood) >=
        woodAfterIgnition) {
        std::cerr << "Fire did not spread through tomb wood\n";
        return 1;
    }

    gunpowder::World oilBurnWorld;
    gunpowder::InputState oilBurnInput;
    oilBurnInput.paint = true;
    oilBurnInput.paintMaterial = gunpowder::Material::fire;
    oilBurnInput.aim = {185.0F, 127.5F};
    oilBurnWorld.update(1.0F / 120.0F, oilBurnInput);
    oilBurnInput.paint = false;
    const std::uint64_t oilAfterIgnition =
        massOf(oilBurnWorld, gunpowder::Material::oil);
    for (int tick = 0; tick < 360; ++tick) {
        oilBurnWorld.update(1.0F / 120.0F, oilBurnInput);
    }
    const std::uint64_t oilAfterBurning =
        massOf(oilBurnWorld, gunpowder::Material::oil);
    if (oilAfterBurning >= oilAfterIgnition || oilAfterBurning == 0) {
        std::cerr << "Oil did not burn gradually as persistent fuel ("
                  << oilAfterIgnition << " -> " << oilAfterBurning
                  << ")\n";
        return 1;
    }

    gunpowder::World smokeWorld;
    gunpowder::InputState smokeInput;
    smokeInput.paint = true;
    smokeInput.paintMaterial = gunpowder::Material::smoke;
    smokeInput.aim = {140.0F, 90.0F};
    smokeWorld.update(1.0F / 120.0F, smokeInput);
    smokeInput.paint = false;
    const std::size_t initialSmoke =
        countOf(smokeWorld, gunpowder::Material::smoke);
    int initialSmokeMinX = gunpowder::World::width;
    int initialSmokeMaxX = 0;
    for (int y = 0; y < gunpowder::World::height; ++y) {
        for (int x = 0; x < gunpowder::World::width; ++x) {
            if (smokeWorld.cell(x, y) == gunpowder::Material::smoke) {
                initialSmokeMinX = std::min(initialSmokeMinX, x);
                initialSmokeMaxX = std::max(initialSmokeMaxX, x);
            }
        }
    }
    for (int tick = 0; tick < 600; ++tick) {
        smokeWorld.update(1.0F / 120.0F, smokeInput);
    }
    int finalSmokeMinX = gunpowder::World::width;
    int finalSmokeMaxX = 0;
    for (int y = 0; y < gunpowder::World::height; ++y) {
        for (int x = 0; x < gunpowder::World::width; ++x) {
            if (smokeWorld.cell(x, y) == gunpowder::Material::smoke) {
                finalSmokeMinX = std::min(finalSmokeMinX, x);
                finalSmokeMaxX = std::max(finalSmokeMaxX, x);
            }
        }
    }
    if (countOf(smokeWorld, gunpowder::Material::smoke) <
        initialSmoke * 3 / 4) {
        std::cerr << "Smoke dissipated too quickly\n";
        return 1;
    }
    if (finalSmokeMaxX - finalSmokeMinX <=
        initialSmokeMaxX - initialSmokeMinX + 4) {
        std::cerr << "Smoke did not spread horizontally\n";
        return 1;
    }

    gunpowder::World grappleWorld;
    gunpowder::InputState grappleInput;
    grappleInput.grappleToggle = true;
    grappleInput.aim = {48.0F, 140.0F};
    grappleWorld.update(1.0F / 120.0F, grappleInput);
    grappleInput.grappleToggle = false;
    for (int tick = 1; tick < 60 && !grappleWorld.grapple().attached; ++tick) {
        grappleWorld.update(1.0F / 120.0F, grappleInput);
    }
    if (!grappleWorld.grapple().attached ||
        grappleWorld.grapple().points.size() < 3) {
        std::cerr << "Grappling hook did not attach and create a chain\n";
        return 1;
    }

    for (int tick = 0; tick < 240 && !grappleWorld.player().onGround; ++tick) {
        grappleWorld.update(1.0F / 120.0F, grappleInput);
    }
    if (!grappleWorld.player().onGround) {
        std::cerr << "Player did not reach the ground while grappled\n";
        return 1;
    }
    grappleInput.jump = true;
    grappleWorld.update(1.0F / 120.0F, grappleInput);
    grappleInput.jump = false;
    if (grappleWorld.player().velocity.y >= 0.0F ||
        !grappleWorld.grapple().attached) {
        std::cerr << "Grounded player could not jump while grappled\n";
        return 1;
    }

    const float originalRopeLength = grappleWorld.grapple().ropeLength;
    grappleInput.reelIn = true;
    for (int tick = 0; tick < 30; ++tick) {
        grappleWorld.update(1.0F / 120.0F, grappleInput);
    }
    if (grappleWorld.grapple().ropeLength >= originalRopeLength) {
        std::cerr << "Grappling hook did not reel in\n";
        return 1;
    }
    grappleInput.grappleToggle = true;
    grappleInput.reelIn = false;
    grappleWorld.update(1.0F / 120.0F, grappleInput);
    if (grappleWorld.grapple().active) {
        std::cerr << "Grappling hook did not release\n";
        return 1;
    }

    gunpowder::World momentumWorld;
    gunpowder::InputState momentumInput;
    momentumInput.grappleToggle = true;
    momentumInput.aim = {48.0F, 30.0F};
    momentumWorld.update(1.0F / 120.0F, momentumInput);
    momentumInput.grappleToggle = false;
    for (int tick = 0; tick < 90 && !momentumWorld.grapple().attached;
         ++tick) {
        momentumWorld.update(1.0F / 120.0F, momentumInput);
    }
    if (!momentumWorld.grapple().attached) {
        std::cerr << "Momentum test hook did not attach overhead\n";
        return 1;
    }

    momentumInput.moveRight = true;
    for (int tick = 0; tick < 90; ++tick) {
        momentumWorld.update(1.0F / 120.0F, momentumInput);
    }
    momentumInput.moveRight = false;
    momentumInput.grappleToggle = true;
    momentumWorld.update(1.0F / 120.0F, momentumInput);
    momentumInput.grappleToggle = false;
    const float releasedHorizontalVelocity =
        momentumWorld.player().velocity.x;
    if (momentumWorld.grapple().active ||
        std::abs(releasedHorizontalVelocity) < 2.0F) {
        std::cerr << "Swing did not produce releasable momentum\n";
        return 1;
    }
    for (int tick = 0; tick < 5; ++tick) {
        momentumWorld.update(1.0F / 120.0F, momentumInput);
    }
    if (std::abs(momentumWorld.player().velocity.x -
                 releasedHorizontalVelocity) > 0.08F) {
        std::cerr << "Horizontal swing momentum was lost after release\n";
        return 1;
    }

    gunpowder::World airControlWorld;
    gunpowder::InputState airControlInput;
    for (int tick = 0;
         tick < 360 && !airControlWorld.player().onGround; ++tick) {
        airControlWorld.update(1.0F / 120.0F, airControlInput);
    }
    if (!airControlWorld.player().onGround) {
        std::cerr << "Air-control test player did not reach the ground\n";
        return 1;
    }
    airControlInput.moveRight = true;
    airControlInput.jump = true;
    airControlWorld.update(1.0F / 120.0F, airControlInput);
    airControlInput.jump = false;
    float maximumAirControlledSpeed =
        std::abs(airControlWorld.player().velocity.x);
    for (int tick = 0;
         tick < 240 && !airControlWorld.player().onGround; ++tick) {
        airControlWorld.update(1.0F / 120.0F, airControlInput);
        maximumAirControlledSpeed =
            std::max(maximumAirControlledSpeed,
                     std::abs(airControlWorld.player().velocity.x));
    }
    if (maximumAirControlledSpeed > 46.01F) {
        std::cerr << "Air controls exceeded maximum run speed\n";
        return 1;
    }

    gunpowder::World unevenGroundWorld;
    constexpr int terrainLeft = 20;
    constexpr int terrainRight = 190;
    constexpr int terrainCeiling = 60;
    constexpr int terrainBottom = 112;
    for (int y = terrainCeiling; y <= terrainBottom; ++y) {
        for (int x = terrainLeft; x <= terrainRight; ++x) {
            unevenGroundWorld.setCellForTest(
                x, y, gunpowder::Material::air);
        }
    }
    constexpr std::array<int, 6> surfaceOffsets{
        0, -1, 0, 1, 0, -1,
    };
    for (int x = terrainLeft; x <= terrainRight; ++x) {
        const int segment = (x - terrainLeft) / 8;
        const int surfaceY =
            92 + surfaceOffsets[static_cast<std::size_t>(
                     segment % static_cast<int>(surfaceOffsets.size()))];
        for (int y = surfaceY; y <= terrainBottom; ++y) {
            unevenGroundWorld.setCellForTest(
                x, y, gunpowder::Material::stone);
        }
    }
    unevenGroundWorld.setPlayerForTest({30.0F, 80.0F});
    gunpowder::InputState unevenGroundInput;
    for (int tick = 0;
         tick < 240 && !unevenGroundWorld.player().onGround; ++tick) {
        unevenGroundWorld.update(1.0F / 120.0F, unevenGroundInput);
    }
    if (!unevenGroundWorld.player().onGround) {
        std::cerr << "Uneven-ground test player did not settle\n";
        return 1;
    }

    unevenGroundInput.moveRight = true;
    int lostGroundFrames = 0;
    for (int tick = 0; tick < 300; ++tick) {
        unevenGroundWorld.update(1.0F / 120.0F, unevenGroundInput);
        if (!unevenGroundWorld.player().onGround) {
            ++lostGroundFrames;
        }
    }
    if (unevenGroundWorld.player().position.x < 125.0F) {
        std::cerr << "Player snagged on one-cell terrain changes at x="
                  << unevenGroundWorld.player().position.x
                  << ", y=" << unevenGroundWorld.player().position.y
                  << ", grounded=" << unevenGroundWorld.player().onGround
                  << ", vx=" << unevenGroundWorld.player().velocity.x
                  << ", hp=" << unevenGroundWorld.player().health
                  << '\n';
        return 1;
    }
    if (lostGroundFrames > 2) {
        std::cerr << "Player repeatedly lost contact with uneven ground\n";
        return 1;
    }

    unevenGroundInput.moveRight = false;
    const float releasePositionX =
        unevenGroundWorld.player().position.x;
    for (int tick = 0; tick < 90; ++tick) {
        unevenGroundWorld.update(1.0F / 120.0F, unevenGroundInput);
    }
    if (std::abs(unevenGroundWorld.player().velocity.x) > 0.5F ||
        unevenGroundWorld.player().position.x - releasePositionX > 8.0F) {
        std::cerr << "Ground friction was unstable on uneven terrain\n";
        return 1;
    }

    gunpowder::World gravityWorld;
    constexpr int gravitySandX = 100;
    constexpr int gravityPlayerX = 125;
    constexpr int gravityStartY = 80;
    for (int y = 30; y < 220; ++y) {
        for (int x = 80; x < 145; ++x) {
            gravityWorld.setCellForTest(
                x, y, gunpowder::Material::air);
        }
    }
    gravityWorld.setCellForTest(
        gravitySandX, gravityStartY, gunpowder::Material::sand);
    gravityWorld.setPlayerForTest(
        {static_cast<float>(gravityPlayerX),
         static_cast<float>(gravityStartY)});
    gunpowder::InputState gravityInput;
    // Twenty-two exact 30 Hz material intervals, expressed as 120 Hz world
    // steps, avoids comparing the player against a partially accumulated
    // material interval.
    for (int tick = 0; tick < 88; ++tick) {
        gravityWorld.update(1.0F / 120.0F, gravityInput);
    }
    int fallenSandY = -1;
    for (int y = gravityStartY; y < 220; ++y) {
        if (gravityWorld.cell(gravitySandX, y) ==
            gunpowder::Material::sand) {
            fallenSandY = y;
            break;
        }
    }
    const float sandFallDistance =
        static_cast<float>(fallenSandY - gravityStartY);
    const float playerFallDistance =
        gravityWorld.player().position.y -
        static_cast<float>(gravityStartY);
    if (fallenSandY < 0 ||
        std::abs(sandFallDistance - playerFallDistance) > 1.5F) {
        std::cerr << "Sand and player gravity diverged: sand="
                  << sandFallDistance << ", player="
                  << playerFallDistance << '\n';
        return 1;
    }

    gunpowder::World firstParallelSandWorld;
    gunpowder::World secondParallelSandWorld;
    for (int y = 30; y <= 150; ++y) {
        for (int x = 52; x <= 76; ++x) {
            const gunpowder::Material material =
                y == 150
                    ? gunpowder::Material::stone
                    : gunpowder::Material::air;
            firstParallelSandWorld.setCellForTest(
                x, y, material);
            secondParallelSandWorld.setCellForTest(
                x, y, material);
        }
    }
    for (int y = 48; y <= 62; ++y) {
        for (int x = 60; x <= 67; ++x) {
            firstParallelSandWorld.setCellForTest(
                x, y, gunpowder::Material::sand);
            secondParallelSandWorld.setCellForTest(
                x, y, gunpowder::Material::sand);
        }
    }
    const std::size_t initialParallelSand =
        countOf(firstParallelSandWorld,
                gunpowder::Material::sand);
    bool usedParallelGranularScheduler = false;
    bool acceptedGranularTransfer = false;
    for (int tick = 0; tick < 60; ++tick) {
        firstParallelSandWorld.update(
            1.0F / 30.0F, gravityInput);
        secondParallelSandWorld.update(
            1.0F / 30.0F, gravityInput);
        const auto& timings =
            firstParallelSandWorld
                .materialSimulationTimings();
        usedParallelGranularScheduler =
            usedParallelGranularScheduler ||
            timings.parallelGranularChunks >= 2;
        acceptedGranularTransfer =
            acceptedGranularTransfer ||
            timings.granularMovesAccepted > 0;
    }
    if (countOf(firstParallelSandWorld,
                gunpowder::Material::sand) !=
        initialParallelSand) {
        std::cerr
            << "Parallel granular transfers changed sand mass\n";
        return 1;
    }
    bool crossedVerticalChunkBoundary = false;
    for (int y = 64; y < 150; ++y) {
        for (int x = 52; x <= 76; ++x) {
            crossedVerticalChunkBoundary =
                crossedVerticalChunkBoundary ||
                firstParallelSandWorld.cell(x, y) ==
                    gunpowder::Material::sand;
        }
    }
    if (!crossedVerticalChunkBoundary ||
        !usedParallelGranularScheduler ||
        !acceptedGranularTransfer) {
        std::cerr
            << "Sand did not use parallel cross-chunk transfers\n";
        return 1;
    }
    if (!std::equal(
            firstParallelSandWorld.materials().begin(),
            firstParallelSandWorld.materials().end(),
            secondParallelSandWorld.materials().begin())) {
        std::cerr
            << "Parallel granular transfers are not deterministic\n";
        return 1;
    }

    gunpowder::World wrapWorld;
    gunpowder::InputState wrapInput;
    wrapInput.grappleToggle = true;
    wrapInput.aim = {155.0F, 140.0F};
    wrapWorld.update(1.0F / 120.0F, wrapInput);
    wrapInput.grappleToggle = false;
    for (int tick = 0; tick < 90 && !wrapWorld.grapple().attached; ++tick) {
        wrapWorld.update(1.0F / 120.0F, wrapInput);
    }
    if (!wrapWorld.grapple().attached) {
        std::cerr << "Terrain wrapping test hook did not attach\n";
        return 1;
    }

    wrapInput.moveRight = true;
    bool observedTerrainWrap = false;
    for (int tick = 0; tick < 720; ++tick) {
        wrapWorld.update(1.0F / 120.0F, wrapInput);
        observedTerrainWrap =
            observedTerrainWrap || !wrapWorld.grapple().wrapPoints.empty();
        if (!wrapWorld.grapple().active || !wrapWorld.grapple().attached) {
            std::cerr << "Grappling hook broke under sustained tension\n";
            return 1;
        }
    }
    if (!observedTerrainWrap) {
        std::cerr << "Rope did not wrap around the platform corner\n";
        return 1;
    }
    return 0;
}
