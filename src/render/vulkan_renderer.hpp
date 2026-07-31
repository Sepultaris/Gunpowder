#pragma once

#include "game/world.hpp"

#include <SDL.h>
#include <vulkan/vulkan.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace gunpowder {

struct Vertex {
    float x;
    float y;
    float r;
    float g;
    float b;
    float a;
    float worldX;
    float worldY;
    float lightingStrength;
};

struct RayTracingSettings {
    int raysPerLight = 16;
    float softShadowRadius = 1.35F * World::simulationScale;
    float solidTransmission = 0.002F;
    float smokeTransmission = 0.94F;
    float steamTransmission = 0.975F;

    int indirectBounces = 2;
    int giRays = 2;
    float giIntensity = 0.38F;
    float bounceDistance = 128.0F * World::simulationScale;
    bool temporalGiDenoising = true;
    float giHistoryWeight = 0.94F;
    bool adaptiveGiSampling = true;
    int debugDisplayMode = 0;

    float playerLightRadius = 45.85F * World::simulationScale;
    float playerLightIntensity = 0.85F;

    float fireBaseRadius = 14.0F * World::simulationScale;
    float fireClusterRadiusScale = 4.0F;
    float fireMaximumRadiusBonus = 20.0F * World::simulationScale;
    float fireBaseIntensity = 0.30F;
    float fireClusterIntensityScale =
        0.08F /
        static_cast<float>(World::simulationScale *
                           World::simulationScale);
    float fireMaximumIntensity = 1.0F;

    bool dayNightCycle = true;
    float timeOfDayHours = 10.0F;
    float dayLengthSeconds = 240.0F;
    int sunRays = 1;
    float sunIntensity = 1.15F;
    float sunShadowSoftness = 0.018F;
    float skyIntensity = 0.90F;
    float skyLightIntensity = 0.16F;
    int skyRays = 5;

    float ambientIntensity = 0.002F;
    float daylightAmbientIntensity = 0.018F;
    float baseHaze = 0.075F;
    float smokeHazeContribution = 0.62F;
    float hazeAttenuation = 0.12F;

    float materialSpecularStrength = 1.0F;
    float materialNormalDetail = 1.0F;
    float marblePolish = 1.0F;
    float marbleVeinReflectivity = 1.0F;
    float marbleFleckDensity = 0.035F;
    float marbleFleckReflectivity = 1.0F;
    float marbleSubsurfaceStrength = 0.72F;
    float marbleScatterDistance = 4.0F;

    bool liquidMetaballs = true;
    float liquidMetaballDensity = 1.0F;
    float liquidMetaballRadius = 0.82F;
    float liquidMetaballEdgeSoftness = 0.085F;
    float liquidFlowStretch = 0.85F;
    float liquidPoolFlattening = 0.62F;
    float liquidMetaballNormalStrength = 1.0F;
    float liquidReflectionStrength = 1.0F;
    float liquidSubsurfaceStrength = 0.68F;
    float liquidCausticStrength = 0.55F;
};

struct GpuRayTimings {
    float derivedFieldsMs = 0.0F;
    float directLightingMs = 0.0F;
    float globalIlluminationMs = 0.0F;
    float denoisingMs = 0.0F;
    float totalMs = 0.0F;
    bool valid = false;
};

class VulkanRenderer {
public:
    explicit VulkanRenderer(SDL_Window* window);
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void draw(World& world);
    void synchronizeMaterialSimulation(World& world);
    void beginUiFrame();
    void refreshDisplay();
    void waitIdle() const;
    [[nodiscard]] RayTracingSettings& rayTracingSettings() {
        return rayTracingSettings_;
    }
    void setPlayerLightEnabled(bool enabled) {
        playerLightEnabled_ = enabled;
    }
    [[nodiscard]] bool playerLightEnabled() const {
        return playerLightEnabled_;
    }
    [[nodiscard]] const GpuRayTimings& gpuRayTimings() const {
        return gpuRayTimings_;
    }

private:
    static constexpr std::size_t framesInFlight = 2;
    static constexpr VkDeviceSize vertexBufferBytes = 16U * 1024U * 1024U;
    static constexpr std::uint32_t textureWidth = World::viewWidth + 2;
    static constexpr std::uint32_t textureHeight = World::viewHeight + 2;
    // Lighting matches the material grid exactly. Display resolution remains
    // independent and can upscale both fields together.
    static constexpr std::uint32_t lightingResolutionScale = 1;
    static constexpr std::uint32_t lightingWidth =
        (textureWidth + lightingResolutionScale - 1) /
        lightingResolutionScale;
    static constexpr std::uint32_t lightingHeight =
        (textureHeight + lightingResolutionScale - 1) /
        lightingResolutionScale;
    static constexpr std::uint32_t maximumGpuParticles = 16'384;
    static constexpr std::size_t maximumFireLights = 64;
    static constexpr std::uint32_t lightTileSize = 8;
    static constexpr std::uint32_t lightTileColumns =
        (lightingWidth + lightTileSize - 1) / lightTileSize;
    static constexpr std::uint32_t lightTileRows =
        (lightingHeight + lightTileSize - 1) / lightTileSize;
    static constexpr std::uint32_t lightTileCount =
        lightTileColumns * lightTileRows;
    static constexpr std::uint32_t occupancyBlockSize = 8;
    static constexpr std::uint32_t occupancyLargeBlockSize = 32;
    static constexpr std::uint32_t occupancyColumns =
        (textureWidth + occupancyBlockSize - 1) /
        occupancyBlockSize;
    static constexpr std::uint32_t occupancyRows =
        (textureHeight + occupancyBlockSize - 1) /
        occupancyBlockSize;
    static constexpr std::uint32_t occupancyLargeColumns =
        (textureWidth + occupancyLargeBlockSize - 1) /
        occupancyLargeBlockSize;
    static constexpr std::uint32_t occupancyLargeRows =
        (textureHeight + occupancyLargeBlockSize - 1) /
        occupancyLargeBlockSize;
    static constexpr std::uint32_t occupancyBlockCount =
        occupancyColumns * occupancyRows;
    static constexpr std::uint32_t occupancyLargeBlockCount =
        occupancyLargeColumns * occupancyLargeRows;
    // Legacy pressure-wave compute storage. The authoritative Noita-style
    // material solver is CPU cellular and does not use this allocation.
    static constexpr std::uint32_t maximumSimulationWidth = 768;
    static constexpr std::uint32_t maximumSimulationHeight = 512;
    static constexpr std::uint32_t maximumSimulationCells =
        maximumSimulationWidth * maximumSimulationHeight;

    struct FrameResources {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkQueryPool rayTimingQueryPool = VK_NULL_HANDLE;
        bool rayTimingQueriesValid = false;
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceMemory vertexMemory = VK_NULL_HANDLE;
        VkBuffer textureStagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory textureStagingMemory = VK_NULL_HANDLE;
        VkImage textureImage = VK_NULL_HANDLE;
        VkDeviceMemory textureMemory = VK_NULL_HANDLE;
        VkImageView textureView = VK_NULL_HANDLE;
        VkImage visibilityImage = VK_NULL_HANDLE;
        VkDeviceMemory visibilityMemory = VK_NULL_HANDLE;
        VkImageView visibilityView = VK_NULL_HANDLE;
        VkImage derivedImage = VK_NULL_HANDLE;
        VkDeviceMemory derivedMemory = VK_NULL_HANDLE;
        VkImageView derivedView = VK_NULL_HANDLE;
        VkImage hazeScratchImage = VK_NULL_HANDLE;
        VkDeviceMemory hazeScratchMemory = VK_NULL_HANDLE;
        VkImageView hazeScratchView = VK_NULL_HANDLE;
        VkImage giIntermediateImage = VK_NULL_HANDLE;
        VkDeviceMemory giIntermediateMemory = VK_NULL_HANDLE;
        VkImageView giIntermediateView = VK_NULL_HANDLE;
        VkImage giFinalImage = VK_NULL_HANDLE;
        VkDeviceMemory giFinalMemory = VK_NULL_HANDLE;
        VkImageView giFinalView = VK_NULL_HANDLE;
        VkBuffer particleSpawnBuffer = VK_NULL_HANDLE;
        VkDeviceMemory particleSpawnMemory = VK_NULL_HANDLE;
        VkBuffer sceneLightBuffer = VK_NULL_HANDLE;
        VkDeviceMemory sceneLightMemory = VK_NULL_HANDLE;
        VkBuffer lightTileBuffer = VK_NULL_HANDLE;
        VkDeviceMemory lightTileMemory = VK_NULL_HANDLE;
        VkBuffer occupancyBuffer = VK_NULL_HANDLE;
        VkDeviceMemory occupancyMemory = VK_NULL_HANDLE;
        std::uint32_t particleSpawnCount = 0;
        VkDescriptorSet textureDescriptor = VK_NULL_HANDLE;
        bool textureInitialized = false;
        bool lightingHistoryValid = false;
        std::int32_t lightingOriginX = 0;
        std::int32_t lightingOriginY = 0;
    };

    struct SpritePixel {
        std::uint8_t x = 0;
        std::uint8_t y = 0;
        std::array<float, 3> color{};
    };

    struct alignas(16) GpuParticle {
        std::array<float, 2> position{};
        std::array<float, 2> velocity{};
        std::array<float, 4> colorAndLife{};
        std::array<float, 2> maximumLifeAndSize{};
        std::uint32_t slot = 0;
        std::uint32_t padding = 0;
    };

    struct alignas(16) GpuMaterialCell {
        std::uint32_t material = 0;
        std::uint32_t liquidAmount = 0;
        std::int32_t flowX = 0;
        std::int32_t flowY = 0;
        float heat = 0.0F;
        float burnProgress = 0.0F;
        float gasLifetime = 0.0F;
        std::int32_t gasDrift = 0;
        float density = 1.0F;
        float momentumX = 0.0F;
        float momentumY = 0.0F;
        float totalEnergy = 2.5F;
    };

    struct alignas(16) GpuSceneLights {
        std::array<float, 4> playerLight{};
        std::array<std::array<float, 4>, maximumFireLights> fireLights{};
        std::array<std::uint32_t, 4> counts{};
    };

    struct GpuLightTile {
        std::uint32_t count = 0;
        std::array<std::uint32_t, maximumFireLights> indices{};
    };

    struct alignas(16) GpuOccupancyHierarchy {
        std::array<std::uint32_t, 4> dimensions{
            occupancyColumns,
            occupancyRows,
            occupancyLargeColumns,
            occupancyLargeRows,
        };
        std::array<std::uint32_t,
                   occupancyBlockCount +
                       occupancyLargeBlockCount>
            occupied{};
    };

    void createInstance();
    void createSurface();
    void choosePhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createRenderPass();
    void createTextureDescriptors();
    void createPipeline();
    void createFramebuffers();
    void createCommands();
    void createFrameResources();
    void initializeImGui();
    void shutdownImGui();
    void loadPlayerSprite();
    void uploadParticleSpawns(FrameResources& frame, World& world);
    void prepareMaterialSimulation(World& world);
    void recreateSwapchain();
    void destroySwapchain();

    [[nodiscard]] std::vector<Vertex> buildVertices(const World& world) const;
    void updateSceneLights(const World& world, Vec2 camera);
    void recordCommands(VkCommandBuffer commandBuffer, std::uint32_t imageIndex,
                        std::uint32_t vertexCount, const World& world);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& memory);
    [[nodiscard]] std::uint32_t findMemoryType(std::uint32_t typeFilter,
                                               VkMemoryPropertyFlags properties) const;
    [[nodiscard]] VkShaderModule loadShader(const char* filename) const;

    SDL_Window* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    std::uint32_t graphicsFamily_ = 0;
    std::uint32_t presentFamily_ = 0;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout textureDescriptorLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool textureDescriptorPool_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkPipelineLayout cellPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline cellPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout computePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline computePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout derivedPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline derivedPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout particleComputePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline particleComputePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout particlePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline particlePipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout materialComputePipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline materialComputePipeline_ = VK_NULL_HANDLE;
    VkImage sunTransmittanceImage_ = VK_NULL_HANDLE;
    VkDeviceMemory sunTransmittanceMemory_ = VK_NULL_HANDLE;
    VkImageView sunTransmittanceView_ = VK_NULL_HANDLE;
    VkBuffer particleBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory particleMemory_ = VK_NULL_HANDLE;
    std::uint32_t particleSpawnCursor_ = 0;
    std::uint64_t lastParticleTicks_ = 0;
    std::uint64_t lastDayCycleTicks_ = 0;
    std::uint64_t lastSunOcclusionTicks_ = 0;
    std::uint64_t lastSkyOcclusionTicks_ = 0;
    std::uint64_t lastSunTransmittanceTicks_ = 0;
    std::uint64_t cachedSunSolidRevision_ =
        std::numeric_limits<std::uint64_t>::max();
    Vec2 cachedSunCamera_{};
    std::uint64_t cachedSkySolidRevision_ =
        std::numeric_limits<std::uint64_t>::max();
    Vec2 cachedSkyCamera_{};
    std::int32_t cachedSkyRayCount_ = 0;
    bool skyVisibilityCacheValid_ = false;
    Vec2 sunTransmittanceDirection_{};
    std::int32_t sunTransmittanceOriginX_ = 0;
    std::int32_t sunTransmittanceOriginY_ = 0;
    float sunTransmittanceSmoke_ = 0.0F;
    float sunTransmittanceSteam_ = 0.0F;
    float sunTransmittanceSoftness_ = 0.0F;
    std::int32_t sunTransmittanceRayCount_ = 0;
    bool sunTransmittanceCacheValid_ = false;
    bool sunTransmittanceImageInitialized_ = false;
    bool updateSunTransmittanceThisFrame_ = false;
    std::uint64_t lastFireLightUpdateTicks_ = 0;
    std::uint64_t lastFireLightSmoothingTicks_ = 0;
    float particleDeltaTime_ = 1.0F / 60.0F;
    bool particleBufferInitialized_ = false;
    VkBuffer materialStateA_ = VK_NULL_HANDLE;
    VkDeviceMemory materialStateMemoryA_ = VK_NULL_HANDLE;
    VkBuffer materialStateB_ = VK_NULL_HANDLE;
    VkDeviceMemory materialStateMemoryB_ = VK_NULL_HANDLE;
    VkBuffer materialUploadBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory materialUploadMemory_ = VK_NULL_HANDLE;
    VkBuffer materialReadbackBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory materialReadbackMemory_ = VK_NULL_HANDLE;
    VkBuffer materialFluxBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory materialFluxMemory_ = VK_NULL_HANDLE;
    std::uint32_t materialSimulationSteps_ = 0;
    std::uint32_t materialSimulationWidth_ = 0;
    std::uint32_t materialSimulationHeight_ = 0;
    std::int32_t materialSimulationOriginX_ = 0;
    std::int32_t materialSimulationOriginY_ = 0;
    std::uint32_t materialSimulationSeed_ = 0;
    bool materialSimulationPending_ = false;
    bool materialSimulationResultInB_ = false;
    std::size_t materialSimulationFrame_ = 0;
    std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::array<FrameResources, framesInFlight> frames_{};
    std::size_t currentFrame_ = 0;
    std::array<float, 4> playerLight_{};
    bool playerLightEnabled_ = true;
    std::array<std::array<float, 4>, maximumFireLights> fireLights_{};
    std::array<std::array<float, 4>, maximumFireLights> fireLightTargets_{};
    RayTracingSettings rayTracingSettings_{};
    RayTracingSettings temporalHistorySettings_{};
    bool temporalHistorySettingsInitialized_ = false;
    std::uint32_t lightingFrameIndex_ = 0;
    std::vector<SpritePixel> playerSprite_;
    std::vector<float> sunHorizonDepths_;
    std::vector<std::int32_t> sunHorizonBlockers_;
    float sunHorizonMinimum_ = 0.0F;
    struct DirectionalHorizon {
        Vec2 direction{};
        std::vector<float> depths;
        std::vector<std::int32_t> blockers;
        float minimumPerpendicularCoordinate = 0.0F;
    };
    std::vector<DirectionalHorizon> skyHorizons_;
    Vec2 cachedSunDirection_{};
    bool sunVisibilityCacheValid_ = false;
    int playerSpriteWidth_ = 0;
    int playerSpriteHeight_ = 0;
    bool imguiInitialized_ = false;
    float timestampPeriodNanoseconds_ = 1.0F;
    GpuRayTimings gpuRayTimings_{};
};

} // namespace gunpowder
