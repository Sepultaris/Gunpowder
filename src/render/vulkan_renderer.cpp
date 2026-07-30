#include "render/vulkan_renderer.hpp"

#include <SDL_vulkan.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace gunpowder {
namespace {

constexpr std::uint32_t cellPushConstantBytes = 128;
constexpr std::uint32_t tracePushConstantBytes = 80;
constexpr std::uint32_t particlePushConstantBytes = 16;
constexpr std::uint32_t materialPushConstantBytes = 32;
constexpr std::uint32_t derivedTimingStart = 0;
constexpr std::uint32_t derivedTimingEnd = 1;
constexpr std::uint32_t directTimingStart = 2;
constexpr std::uint32_t directTimingEnd = 3;
constexpr std::uint32_t giTimingStart = 4;
constexpr std::uint32_t giTimingEnd = 5;
constexpr std::uint32_t denoiseTimingStart = 6;
constexpr std::uint32_t denoiseTimingEnd = 7;
constexpr std::uint32_t rayTimingStart = 8;
constexpr std::uint32_t rayTimingEnd = 9;
constexpr std::uint32_t rayTimingQueryCount = 10;
constexpr float renderScale = static_cast<float>(World::simulationScale);
constexpr float tau = 6.28318530718F;
constexpr float sunHorizonSamplesPerCell = 4.0F;

float packNormalizedPair(
    float first, float firstMaximum,
    float second, float secondMaximum) {
    const auto encode = [](float value, float maximum) {
        return static_cast<std::uint32_t>(std::lround(
            std::clamp(value / std::max(maximum, 0.0001F),
                       0.0F, 1.0F) *
            65535.0F));
    };
    const std::uint32_t packed =
        encode(first, firstMaximum) |
        (encode(second, secondMaximum) << 16U);
    return std::bit_cast<float>(packed);
}

struct CelestialState {
    Vec2 sunDirection{};
    float daylight = 0.0F;
    float daylightFactor = 0.0F;
    float phase = 0.0F;
};

CelestialState celestialState(const RayTracingSettings& settings) {
    const float wrappedHours =
        std::fmod(std::max(settings.timeOfDayHours, 0.0F), 24.0F);
    const float phase = wrappedHours / 24.0F;
    const float angle = (phase - 0.25F) * tau;
    const float sunHeight = std::sin(angle);
    const float daylightBlend = std::clamp(
        (sunHeight + 0.10F) / 0.20F, 0.0F, 1.0F);
    const float smoothDaylight =
        daylightBlend * daylightBlend *
        (3.0F - 2.0F * daylightBlend);
    return {
        .sunDirection = {std::cos(angle), -sunHeight},
        .daylight =
            smoothDaylight * settings.sunIntensity,
        .daylightFactor = smoothDaylight,
        .phase = phase,
    };
}

bool isSunOccluder(Material material) {
    return material == Material::dirt ||
           material == Material::sand ||
           material == Material::rock ||
           material == Material::wood ||
           material == Material::stone ||
           material == Material::metal;
}

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed with Vulkan error " +
                                 std::to_string(result));
    }
}

void checkImGuiVkResult(VkResult result) {
    check(result, "Dear ImGui Vulkan backend");
}

struct QueueFamilies {
    std::uint32_t graphics = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t present = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] bool complete() const {
        return graphics != std::numeric_limits<std::uint32_t>::max() &&
               present != std::numeric_limits<std::uint32_t>::max();
    }
};

QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, properties.data());

    QueueFamilies result;
    for (std::uint32_t index = 0; index < count; ++index) {
        constexpr VkQueueFlags requiredGraphicsFlags =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((properties[index].queueFlags & requiredGraphicsFlags) ==
            requiredGraphicsFlags) {
            result.graphics = index;
        }
        VkBool32 supportsPresent = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &supportsPresent);
        if (supportsPresent == VK_TRUE) {
            result.present = index;
        }
        if (result.complete()) {
            break;
        }
    }
    return result;
}

bool supportsSwapchain(VkPhysicalDevice device, VkSurfaceKHR surface) {
    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         extensions.data());
    const bool hasExtension = std::ranges::any_of(extensions, [](const auto& extension) {
        return std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
    });
    if (!hasExtension) {
        return false;
    }

    std::uint32_t formatCount = 0;
    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    return formatCount > 0 && presentModeCount > 0;
}

bool supportsLightingStorageImage(VkPhysicalDevice device) {
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceFeatures(device, &features);
    if (features.shaderStorageImageExtendedFormats != VK_TRUE) {
        return false;
    }
    VkFormatProperties integerProperties{};
    vkGetPhysicalDeviceFormatProperties(
        device, VK_FORMAT_R8G8B8A8_UINT, &integerProperties);
    VkFormatProperties giProperties{};
    vkGetPhysicalDeviceFormatProperties(
        device, VK_FORMAT_R16G16B16A16_SFLOAT, &giProperties);
    return (integerProperties.optimalTilingFeatures &
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0U &&
           (giProperties.optimalTilingFeatures &
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0U;
}

void appendQuad(std::vector<Vertex>& vertices, float left, float top, float right,
                float bottom, std::array<float, 3> color, float alpha = 1.0F) {
    const Vertex topLeft{
        left, top, color[0], color[1], color[2], alpha, 0.0F, 0.0F, 0.0F};
    const Vertex topRight{
        right, top, color[0], color[1], color[2], alpha, 0.0F, 0.0F, 0.0F};
    const Vertex bottomLeft{
        left, bottom, color[0], color[1], color[2], alpha, 0.0F, 0.0F, 0.0F};
    const Vertex bottomRight{
        right, bottom, color[0], color[1], color[2], alpha, 0.0F, 0.0F, 0.0F};
    vertices.insert(vertices.end(), {
        topLeft, bottomLeft, bottomRight,
        topLeft, bottomRight, topRight,
    });
}

std::array<float, 3> materialColor(Material material, int x, int y) {
    const float variation = static_cast<float>((x * 13 + y * 7) & 7) * 0.008F;
    switch (material) {
    case Material::dirt:
        return {0.39F + variation, 0.23F + variation, 0.11F};
    case Material::sand:
        return {0.76F + variation, 0.59F + variation, 0.23F};
    case Material::rock:
        return {0.24F + variation, 0.25F + variation, 0.28F + variation};
    case Material::water:
        return {0.08F, 0.36F + variation, 0.82F + variation};
    case Material::oil:
        return {0.20F + variation, 0.12F, 0.25F + variation};
    case Material::fire:
        return {1.0F, 0.28F + variation * 4.0F, 0.035F};
    case Material::smoke:
        return {0.32F + variation, 0.34F + variation, 0.39F + variation};
    case Material::steam:
        return {0.68F + variation, 0.76F + variation, 0.82F + variation};
    case Material::wood:
        return {0.46F + variation, 0.24F + variation * 0.5F, 0.075F};
    case Material::stone:
        return {0.48F + variation, 0.43F + variation,
                0.32F + variation * 0.6F};
    case Material::metal:
        return {0.34F + variation, 0.39F + variation,
                0.43F + variation};
    case Material::air:
        break;
    }
    return {};
}

} // namespace

VulkanRenderer::VulkanRenderer(SDL_Window* window) : window_(window) {
    loadPlayerSprite();
    createInstance();
    createSurface();
    choosePhysicalDevice();
    createDevice();
    createSwapchain();
    createRenderPass();
    createTextureDescriptors();
    createPipeline();
    createFramebuffers();
    createCommands();
    createFrameResources();
    initializeImGui();
}

void VulkanRenderer::loadPlayerSprite() {
    char* basePath = SDL_GetBasePath();
    if (basePath == nullptr) {
        throw std::runtime_error(SDL_GetError());
    }
    const std::filesystem::path path =
        std::filesystem::path(basePath) / "sprites" /
        "player_hooded.bmp";
    SDL_free(basePath);

    SDL_Surface* loaded = SDL_LoadBMP(path.string().c_str());
    if (loaded == nullptr) {
        throw std::runtime_error("Could not load player sprite: " +
                                 path.string() + " (" + SDL_GetError() + ")");
    }
    SDL_Surface* surface =
        SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded);
    if (surface == nullptr) {
        throw std::runtime_error(
            "Could not convert player sprite: " +
            std::string(SDL_GetError()));
    }

    playerSpriteWidth_ = surface->w;
    playerSpriteHeight_ = surface->h;
    playerSprite_.clear();
    playerSprite_.reserve(
        static_cast<std::size_t>(surface->w * surface->h));
    const auto* pixels =
        static_cast<const std::uint8_t*>(surface->pixels);
    for (int y = 0; y < surface->h; ++y) {
        for (int x = 0; x < surface->w; ++x) {
            std::uint32_t pixel = 0;
            std::memcpy(
                &pixel,
                pixels + static_cast<std::size_t>(y * surface->pitch) +
                    static_cast<std::size_t>(x * 4),
                sizeof(pixel));
            std::uint8_t red = 0;
            std::uint8_t green = 0;
            std::uint8_t blue = 0;
            std::uint8_t alpha = 0;
            SDL_GetRGBA(pixel, surface->format, &red, &green, &blue, &alpha);
            const bool chromaKey =
                red >= 248 && green <= 7 && blue >= 248;
            if (chromaKey || alpha < 96) {
                continue;
            }
            playerSprite_.push_back(SpritePixel{
                .x = static_cast<std::uint8_t>(x),
                .y = static_cast<std::uint8_t>(y),
                .color = {
                    static_cast<float>(red) / 255.0F,
                    static_cast<float>(green) / 255.0F,
                    static_cast<float>(blue) / 255.0F,
                },
            });
        }
    }
    SDL_FreeSurface(surface);
    if (playerSprite_.empty()) {
        throw std::runtime_error("Player sprite contains no visible pixels");
    }
}

VulkanRenderer::~VulkanRenderer() {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
        shutdownImGui();
        for (FrameResources& frame : frames_) {
            if (frame.vertexBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.vertexBuffer, nullptr);
            }
            if (frame.vertexMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.vertexMemory, nullptr);
            }
            if (frame.textureView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.textureView, nullptr);
            }
            if (frame.textureImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.textureImage, nullptr);
            }
            if (frame.textureMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.textureMemory, nullptr);
            }
            if (frame.visibilityView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.visibilityView, nullptr);
            }
            if (frame.visibilityImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.visibilityImage, nullptr);
            }
            if (frame.visibilityMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.visibilityMemory, nullptr);
            }
            if (frame.derivedView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.derivedView, nullptr);
            }
            if (frame.derivedImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.derivedImage, nullptr);
            }
            if (frame.derivedMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.derivedMemory, nullptr);
            }
            if (frame.hazeScratchView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.hazeScratchView, nullptr);
            }
            if (frame.hazeScratchImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.hazeScratchImage, nullptr);
            }
            if (frame.hazeScratchMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.hazeScratchMemory, nullptr);
            }
            if (frame.giIntermediateView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.giIntermediateView, nullptr);
            }
            if (frame.giIntermediateImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.giIntermediateImage, nullptr);
            }
            if (frame.giIntermediateMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.giIntermediateMemory, nullptr);
            }
            if (frame.giFinalView != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, frame.giFinalView, nullptr);
            }
            if (frame.giFinalImage != VK_NULL_HANDLE) {
                vkDestroyImage(device_, frame.giFinalImage, nullptr);
            }
            if (frame.giFinalMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.giFinalMemory, nullptr);
            }
            if (frame.particleSpawnBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.particleSpawnBuffer, nullptr);
            }
            if (frame.particleSpawnMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.particleSpawnMemory, nullptr);
            }
            if (frame.sceneLightBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.sceneLightBuffer, nullptr);
            }
            if (frame.sceneLightMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.sceneLightMemory, nullptr);
            }
            if (frame.lightTileBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.lightTileBuffer, nullptr);
            }
            if (frame.lightTileMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.lightTileMemory, nullptr);
            }
            if (frame.occupancyBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.occupancyBuffer, nullptr);
            }
            if (frame.occupancyMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.occupancyMemory, nullptr);
            }
            if (frame.textureStagingBuffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, frame.textureStagingBuffer, nullptr);
            }
            if (frame.textureStagingMemory != VK_NULL_HANDLE) {
                vkFreeMemory(device_, frame.textureStagingMemory, nullptr);
            }
            if (frame.imageAvailable != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, frame.imageAvailable, nullptr);
            }
            if (frame.renderFinished != VK_NULL_HANDLE) {
                vkDestroySemaphore(device_, frame.renderFinished, nullptr);
            }
            if (frame.inFlight != VK_NULL_HANDLE) {
                vkDestroyFence(device_, frame.inFlight, nullptr);
            }
        }
        if (sunTransmittanceView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, sunTransmittanceView_, nullptr);
        }
        if (sunTransmittanceImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, sunTransmittanceImage_, nullptr);
        }
        if (sunTransmittanceMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, sunTransmittanceMemory_, nullptr);
        }
        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }
        if (particleBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, particleBuffer_, nullptr);
        }
        if (particleMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, particleMemory_, nullptr);
        }
        if (materialStateA_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialStateA_, nullptr);
        }
        if (materialStateMemoryA_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, materialStateMemoryA_, nullptr);
        }
        if (materialStateB_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialStateB_, nullptr);
        }
        if (materialStateMemoryB_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, materialStateMemoryB_, nullptr);
        }
        if (materialUploadBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialUploadBuffer_, nullptr);
        }
        if (materialUploadMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, materialUploadMemory_, nullptr);
        }
        if (materialReadbackBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialReadbackBuffer_, nullptr);
        }
        if (materialReadbackMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, materialReadbackMemory_, nullptr);
        }
        if (materialFluxBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, materialFluxBuffer_, nullptr);
        }
        if (materialFluxMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, materialFluxMemory_, nullptr);
        }
        destroySwapchain();
        if (textureDescriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, textureDescriptorPool_, nullptr);
        }
        if (textureSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, textureSampler_, nullptr);
        }
        if (textureDescriptorLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, textureDescriptorLayout_, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanRenderer::initializeImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = "gunpowder_imgui.ini";
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0F;
    style.FrameRounding = 3.0F;

    if (!ImGui_ImplSDL2_InitForVulkan(window_)) {
        ImGui::DestroyContext();
        throw std::runtime_error("Dear ImGui SDL backend initialization failed");
    }

    if (swapchainImages_.size() < 2) {
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        throw std::runtime_error("Dear ImGui requires at least two swapchain images");
    }
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_1;
    initInfo.Instance = instance_;
    initInfo.PhysicalDevice = physicalDevice_;
    initInfo.Device = device_;
    initInfo.QueueFamily = graphicsFamily_;
    initInfo.Queue = graphicsQueue_;
    initInfo.DescriptorPoolSize = 32;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount =
        static_cast<std::uint32_t>(swapchainImages_.size());
    initInfo.PipelineInfoMain.RenderPass = renderPass_;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = checkImGuiVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();
        throw std::runtime_error(
            "Dear ImGui Vulkan backend initialization failed");
    }
    imguiInitialized_ = true;
}

void VulkanRenderer::shutdownImGui() {
    if (!imguiInitialized_) {
        return;
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    imguiInitialized_ = false;
}

void VulkanRenderer::beginUiFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
}

void VulkanRenderer::createInstance() {
    unsigned int extensionCount = 0;
    if (SDL_Vulkan_GetInstanceExtensions(window_, &extensionCount, nullptr) != SDL_TRUE) {
        throw std::runtime_error(SDL_GetError());
    }
    std::vector<const char*> extensions(extensionCount);
    if (SDL_Vulkan_GetInstanceExtensions(window_, &extensionCount, extensions.data()) !=
        SDL_TRUE) {
        throw std::runtime_error(SDL_GetError());
    }

    const VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "Gunpowder",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "Gunpowder",
        .engineVersion = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    check(vkCreateInstance(&createInfo, nullptr, &instance_), "vkCreateInstance");
}

void VulkanRenderer::createSurface() {
    if (SDL_Vulkan_CreateSurface(window_, instance_, &surface_) != SDL_TRUE) {
        throw std::runtime_error(SDL_GetError());
    }
}

void VulkanRenderer::choosePhysicalDevice() {
    std::uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr),
          "vkEnumeratePhysicalDevices");
    if (count == 0) {
        throw std::runtime_error("No Vulkan-capable GPU was found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()),
          "vkEnumeratePhysicalDevices");

    for (VkPhysicalDevice candidate : devices) {
        const QueueFamilies families = findQueueFamilies(candidate, surface_);
        if (families.complete() &&
            supportsSwapchain(candidate, surface_) &&
            supportsLightingStorageImage(candidate)) {
            physicalDevice_ = candidate;
            graphicsFamily_ = families.graphics;
            presentFamily_ = families.present;
            VkPhysicalDeviceProperties properties{};
            vkGetPhysicalDeviceProperties(candidate, &properties);
            timestampPeriodNanoseconds_ =
                properties.limits.timestampPeriod;
            return;
        }
    }
    throw std::runtime_error("No GPU supports the required Vulkan queues and swapchain");
}

void VulkanRenderer::createDevice() {
    const std::set<std::uint32_t> uniqueFamilies{graphicsFamily_, presentFamily_};
    constexpr float priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (std::uint32_t family : uniqueFamilies) {
        queueInfos.push_back({
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queueFamilyIndex = family,
            .queueCount = 1,
            .pQueuePriorities = &priority,
        });
    }
    constexpr std::array extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    VkPhysicalDeviceFeatures features{};
    features.shaderStorageImageExtendedFormats = VK_TRUE;
    const VkDeviceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size()),
        .pQueueCreateInfos = queueInfos.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = static_cast<std::uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
        .pEnabledFeatures = &features,
    };
    check(vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_),
          "vkCreateDevice");
    vkGetDeviceQueue(device_, graphicsFamily_, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, presentFamily_, 0, &presentQueue_);
}

void VulkanRenderer::createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_, surface_,
                                                     &capabilities),
          "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_, surface_, &formatCount,
                                         formats.data());
    VkSurfaceFormatKHR selectedFormat = formats.front();
    for (const VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            selectedFormat = format;
            break;
        }
    }

    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_,
                                              &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_, surface_,
                                              &presentModeCount, presentModes.data());
    VkPresentModeKHR selectedPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    if (std::ranges::find(presentModes, VK_PRESENT_MODE_MAILBOX_KHR) !=
        presentModes.end()) {
        selectedPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    }

    if (capabilities.currentExtent.width !=
        std::numeric_limits<std::uint32_t>::max()) {
        swapchainExtent_ = capabilities.currentExtent;
    } else {
        int width = 0;
        int height = 0;
        SDL_Vulkan_GetDrawableSize(window_, &width, &height);
        swapchainExtent_ = {
            std::clamp(static_cast<std::uint32_t>(width),
                       capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width),
            std::clamp(static_cast<std::uint32_t>(height),
                       capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height),
        };
    }

    std::uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        imageCount = std::min(imageCount, capabilities.maxImageCount);
    }
    const std::array families{graphicsFamily_, presentFamily_};
    const bool separateQueues = graphicsFamily_ != presentFamily_;
    const VkSwapchainCreateInfoKHR createInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .surface = surface_,
        .minImageCount = imageCount,
        .imageFormat = selectedFormat.format,
        .imageColorSpace = selectedFormat.colorSpace,
        .imageExtent = swapchainExtent_,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = separateQueues ? VK_SHARING_MODE_CONCURRENT
                                           : VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = separateQueues ? 2U : 0U,
        .pQueueFamilyIndices = separateQueues ? families.data() : nullptr,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = selectedPresentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    check(vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_),
          "vkCreateSwapchainKHR");

    swapchainFormat_ = selectedFormat.format;
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());

    swapchainImageViews_.resize(swapchainImages_.size());
    for (std::size_t index = 0; index < swapchainImages_.size(); ++index) {
        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = swapchainImages_[index],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = swapchainFormat_,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        check(vkCreateImageView(device_, &viewInfo, nullptr,
                                &swapchainImageViews_[index]),
              "vkCreateImageView");
    }
}

void VulkanRenderer::createRenderPass() {
    const VkAttachmentDescription colorAttachment{
        .flags = 0,
        .format = swapchainFormat_,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    const VkAttachmentReference colorReference{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    const VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorReference,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = nullptr,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
    const VkSubpassDependency dependency{
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };
    const VkRenderPassCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };
    check(vkCreateRenderPass(device_, &createInfo, nullptr, &renderPass_),
          "vkCreateRenderPass");
}

void VulkanRenderer::createTextureDescriptors() {
    const std::array bindings{
        VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 5,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 6,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 7,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 8,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 9,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 10,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 11,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 12,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 13,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_COMPUTE_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 14,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 15,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 16,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 17,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = 18,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        },
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = static_cast<std::uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr,
                                      &textureDescriptorLayout_),
          "vkCreateDescriptorSetLayout");

    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0F,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0F,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0F,
        .maxLod = 0.0F,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
    check(vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_),
          "vkCreateSampler");

    const std::array poolSizes{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount =
                static_cast<std::uint32_t>(framesInFlight * 5),
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount =
                static_cast<std::uint32_t>(framesInFlight * 6),
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount =
                static_cast<std::uint32_t>(framesInFlight * 8),
        },
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = static_cast<std::uint32_t>(framesInFlight),
        .poolSizeCount = static_cast<std::uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    check(vkCreateDescriptorPool(device_, &poolInfo, nullptr,
                                 &textureDescriptorPool_),
          "vkCreateDescriptorPool");
}

VkShaderModule VulkanRenderer::loadShader(const char* filename) const {
    char* basePath = SDL_GetBasePath();
    if (basePath == nullptr) {
        throw std::runtime_error(SDL_GetError());
    }
    const std::filesystem::path path =
        std::filesystem::path(basePath) / "shaders" / filename;
    SDL_free(basePath);

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Could not open shader: " + path.string());
    }
    const std::streamsize size = stream.tellg();
    std::vector<std::uint32_t> code(
        (static_cast<std::size_t>(size) + sizeof(std::uint32_t) - 1) /
        sizeof(std::uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(code.data()), size);

    const VkShaderModuleCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = static_cast<std::size_t>(size),
        .pCode = code.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    check(vkCreateShaderModule(device_, &createInfo, nullptr, &module),
          "vkCreateShaderModule");
    return module;
}

void VulkanRenderer::createPipeline() {
    const VkShaderModule vertexShader = loadShader("world.vert.spv");
    const VkShaderModule fragmentShader = loadShader("world.frag.spv");
    const std::array shaderStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertexShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragmentShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };
    const VkVertexInputBindingDescription binding{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    const std::array attributes{
        VkVertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(Vertex, x),
        },
        VkVertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(Vertex, r),
        },
        VkVertexInputAttributeDescription{
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(Vertex, worldX),
        },
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount =
            static_cast<std::uint32_t>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data(),
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };
    const VkPipelineRasterizationStateCreateInfo rasterizer{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0F,
        .depthBiasClamp = 0.0F,
        .depthBiasSlopeFactor = 0.0F,
        .lineWidth = 1.0F,
    };
    const VkPipelineMultisampleStateCreateInfo multisampling{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0F,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };
    const VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable = VK_TRUE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
        .blendConstants = {},
    };
    constexpr std::array dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<std::uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    const VkPushConstantRange overlayPushRange{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = cellPushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &overlayPushRange,
    };
    check(vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipelineLayout_),
          "vkCreatePipelineLayout");

    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<std::uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = pipelineLayout_,
        .renderPass = renderPass_,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult result = vkCreateGraphicsPipelines(
        device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, fragmentShader, nullptr);
    vkDestroyShaderModule(device_, vertexShader, nullptr);
    check(result, "vkCreateGraphicsPipelines");

    const VkShaderModule cellVertexShader = loadShader("cells.vert.spv");
    const VkShaderModule cellFragmentShader = loadShader("cells.frag.spv");
    const std::array cellShaderStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = cellVertexShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = cellFragmentShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };
    const VkPipelineVertexInputStateCreateInfo emptyVertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 0,
        .pVertexBindingDescriptions = nullptr,
        .vertexAttributeDescriptionCount = 0,
        .pVertexAttributeDescriptions = nullptr,
    };
    const VkPipelineColorBlendAttachmentState opaqueAttachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo opaqueBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &opaqueAttachment,
        .blendConstants = {},
    };
    const VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = cellPushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo cellLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange,
    };
    check(vkCreatePipelineLayout(device_, &cellLayoutInfo, nullptr,
                                 &cellPipelineLayout_),
          "vkCreatePipelineLayout");
    const VkGraphicsPipelineCreateInfo cellPipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<std::uint32_t>(cellShaderStages.size()),
        .pStages = cellShaderStages.data(),
        .pVertexInputState = &emptyVertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &opaqueBlend,
        .pDynamicState = &dynamicState,
        .layout = cellPipelineLayout_,
        .renderPass = renderPass_,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult cellResult = vkCreateGraphicsPipelines(
        device_, VK_NULL_HANDLE, 1, &cellPipelineInfo, nullptr, &cellPipeline_);
    vkDestroyShaderModule(device_, cellFragmentShader, nullptr);
    vkDestroyShaderModule(device_, cellVertexShader, nullptr);
    check(cellResult, "vkCreateGraphicsPipelines");

    const VkShaderModule computeShader =
        loadShader("lighting.comp.spv");
    const VkPushConstantRange computePushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = tracePushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo computeLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &computePushRange,
    };
    check(vkCreatePipelineLayout(device_, &computeLayoutInfo, nullptr,
                                 &computePipelineLayout_),
          "vkCreatePipelineLayout");
    const VkPipelineShaderStageCreateInfo computeStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = computeShader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    const VkComputePipelineCreateInfo computeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = computeStage,
        .layout = computePipelineLayout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult computeResult = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &computeInfo, nullptr,
        &computePipeline_);
    vkDestroyShaderModule(device_, computeShader, nullptr);
    check(computeResult, "vkCreateComputePipelines");

    const VkShaderModule derivedShader =
        loadShader("derived_fields.comp.spv");
    const VkPushConstantRange derivedPushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(std::uint32_t),
    };
    const VkPipelineLayoutCreateInfo derivedLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &derivedPushRange,
    };
    check(vkCreatePipelineLayout(device_, &derivedLayoutInfo, nullptr,
                                 &derivedPipelineLayout_),
          "vkCreatePipelineLayout");
    const VkPipelineShaderStageCreateInfo derivedStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = derivedShader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    const VkComputePipelineCreateInfo derivedInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = derivedStage,
        .layout = derivedPipelineLayout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult derivedResult = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &derivedInfo, nullptr,
        &derivedPipeline_);
    vkDestroyShaderModule(device_, derivedShader, nullptr);
    check(derivedResult, "vkCreateComputePipelines");

    const VkShaderModule particleComputeShader =
        loadShader("particles.comp.spv");
    const VkPushConstantRange particleComputePushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = particlePushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo particleComputeLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &particleComputePushRange,
    };
    check(vkCreatePipelineLayout(device_, &particleComputeLayoutInfo, nullptr,
                                 &particleComputePipelineLayout_),
          "vkCreatePipelineLayout");
    const VkPipelineShaderStageCreateInfo particleComputeStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = particleComputeShader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    const VkComputePipelineCreateInfo particleComputeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = particleComputeStage,
        .layout = particleComputePipelineLayout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult particleComputeResult = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &particleComputeInfo, nullptr,
        &particleComputePipeline_);
    vkDestroyShaderModule(device_, particleComputeShader, nullptr);
    check(particleComputeResult, "vkCreateComputePipelines");

    const VkShaderModule particleVertexShader =
        loadShader("particles.vert.spv");
    const VkShaderModule particleFragmentShader =
        loadShader("world.frag.spv");
    const std::array particleShaderStages{
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = particleVertexShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = particleFragmentShader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    };
    const VkPushConstantRange particleGraphicsPushRange{
        .stageFlags =
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = 0,
        .size = cellPushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo particleLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &particleGraphicsPushRange,
    };
    check(vkCreatePipelineLayout(device_, &particleLayoutInfo, nullptr,
                                 &particlePipelineLayout_),
          "vkCreatePipelineLayout");
    const VkGraphicsPipelineCreateInfo particlePipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount =
            static_cast<std::uint32_t>(particleShaderStages.size()),
        .pStages = particleShaderStages.data(),
        .pVertexInputState = &emptyVertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pTessellationState = nullptr,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = particlePipelineLayout_,
        .renderPass = renderPass_,
        .subpass = 0,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult particleResult = vkCreateGraphicsPipelines(
        device_, VK_NULL_HANDLE, 1, &particlePipelineInfo, nullptr,
        &particlePipeline_);
    vkDestroyShaderModule(device_, particleFragmentShader, nullptr);
    vkDestroyShaderModule(device_, particleVertexShader, nullptr);
    check(particleResult, "vkCreateGraphicsPipelines");

    const VkShaderModule materialComputeShader =
        loadShader("materials.comp.spv");
    const VkPushConstantRange materialComputePushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = materialPushConstantBytes,
    };
    const VkPipelineLayoutCreateInfo materialComputeLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &textureDescriptorLayout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &materialComputePushRange,
    };
    check(vkCreatePipelineLayout(device_, &materialComputeLayoutInfo, nullptr,
                                 &materialComputePipelineLayout_),
          "vkCreatePipelineLayout");
    const VkPipelineShaderStageCreateInfo materialComputeStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = materialComputeShader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };
    const VkComputePipelineCreateInfo materialComputeInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = materialComputeStage,
        .layout = materialComputePipelineLayout_,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    const VkResult materialComputeResult = vkCreateComputePipelines(
        device_, VK_NULL_HANDLE, 1, &materialComputeInfo, nullptr,
        &materialComputePipeline_);
    vkDestroyShaderModule(device_, materialComputeShader, nullptr);
    check(materialComputeResult, "vkCreateComputePipelines");
}

void VulkanRenderer::createFramebuffers() {
    framebuffers_.resize(swapchainImageViews_.size());
    for (std::size_t index = 0; index < swapchainImageViews_.size(); ++index) {
        const VkImageView attachment = swapchainImageViews_[index];
        const VkFramebufferCreateInfo createInfo{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = renderPass_,
            .attachmentCount = 1,
            .pAttachments = &attachment,
            .width = swapchainExtent_.width,
            .height = swapchainExtent_.height,
            .layers = 1,
        };
        check(vkCreateFramebuffer(device_, &createInfo, nullptr,
                                  &framebuffers_[index]),
              "vkCreateFramebuffer");
    }
}

void VulkanRenderer::createCommands() {
    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphicsFamily_,
    };
    check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_),
          "vkCreateCommandPool");

    std::array<VkCommandBuffer, framesInFlight> commandBuffers{};
    const VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = commandPool_,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = static_cast<std::uint32_t>(commandBuffers.size()),
    };
    check(vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers.data()),
          "vkAllocateCommandBuffers");
    for (std::size_t index = 0; index < frames_.size(); ++index) {
        frames_[index].commandBuffer = commandBuffers[index];
    }
}

void VulkanRenderer::createFrameResources() {
    constexpr VkDeviceSize textureBytes =
        static_cast<VkDeviceSize>(textureWidth) *
        static_cast<VkDeviceSize>(textureHeight) * 4U;
    constexpr VkDeviceSize particleBytes =
        static_cast<VkDeviceSize>(maximumGpuParticles) *
        sizeof(GpuParticle);
    constexpr VkDeviceSize materialBytes =
        static_cast<VkDeviceSize>(maximumSimulationCells) *
        sizeof(GpuMaterialCell);
    constexpr VkDeviceSize materialFluxBytes =
        static_cast<VkDeviceSize>(maximumSimulationCells) * 32U;
    static_assert(sizeof(Material) == 1);
    static_assert(sizeof(GpuParticle) == 48);
    static_assert(sizeof(GpuMaterialCell) == 48);
    static_assert(sizeof(GpuSceneLights) ==
                  (maximumFireLights + 2) * 16);
    static_assert(sizeof(GpuLightTile) ==
                  (maximumFireLights + 1) *
                      sizeof(std::uint32_t));

    createBuffer(particleBytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 particleBuffer_, particleMemory_);
    createBuffer(materialBytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 materialStateA_, materialStateMemoryA_);
    createBuffer(materialBytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 materialStateB_, materialStateMemoryB_);
    createBuffer(materialBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 materialUploadBuffer_, materialUploadMemory_);
    createBuffer(materialBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 materialReadbackBuffer_, materialReadbackMemory_);
    createBuffer(materialFluxBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 materialFluxBuffer_, materialFluxMemory_);

    const VkImageCreateInfo sunTransmittanceImageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .extent = {
            lightingWidth,
            lightingHeight,
            1,
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    check(vkCreateImage(device_, &sunTransmittanceImageInfo, nullptr,
                        &sunTransmittanceImage_),
          "vkCreateImage sun transmittance");
    VkMemoryRequirements sunTransmittanceRequirements{};
    vkGetImageMemoryRequirements(device_, sunTransmittanceImage_,
                                 &sunTransmittanceRequirements);
    const VkMemoryAllocateInfo sunTransmittanceAllocation{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = sunTransmittanceRequirements.size,
        .memoryTypeIndex = findMemoryType(
            sunTransmittanceRequirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    check(vkAllocateMemory(device_, &sunTransmittanceAllocation, nullptr,
                           &sunTransmittanceMemory_),
          "vkAllocateMemory sun transmittance");
    check(vkBindImageMemory(device_, sunTransmittanceImage_,
                            sunTransmittanceMemory_, 0),
          "vkBindImageMemory sun transmittance");
    const VkImageViewCreateInfo sunTransmittanceViewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = sunTransmittanceImage_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .components = {},
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    check(vkCreateImageView(device_, &sunTransmittanceViewInfo, nullptr,
                            &sunTransmittanceView_),
          "vkCreateImageView sun transmittance");

    for (FrameResources& frame : frames_) {
            if (frame.rayTimingQueryPool != VK_NULL_HANDLE) {
                vkDestroyQueryPool(device_, frame.rayTimingQueryPool,
                                   nullptr);
            }
        const VkSemaphoreCreateInfo semaphoreInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        };
        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                                &frame.imageAvailable),
              "vkCreateSemaphore");
        check(vkCreateSemaphore(device_, &semaphoreInfo, nullptr,
                                &frame.renderFinished),
              "vkCreateSemaphore");
        check(vkCreateFence(device_, &fenceInfo, nullptr, &frame.inFlight),
              "vkCreateFence");
        const VkQueryPoolCreateInfo timingQueryInfo{
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 10,
            .pipelineStatistics = 0,
        };
        check(vkCreateQueryPool(device_, &timingQueryInfo, nullptr,
                                &frame.rayTimingQueryPool),
              "vkCreateQueryPool");

        createBuffer(vertexBufferBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     frame.vertexBuffer, frame.vertexMemory);
        createBuffer(textureBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     frame.textureStagingBuffer, frame.textureStagingMemory);
        createBuffer(particleBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     frame.particleSpawnBuffer, frame.particleSpawnMemory);
        createBuffer(sizeof(GpuSceneLights),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     frame.sceneLightBuffer, frame.sceneLightMemory);
        createBuffer(
            static_cast<VkDeviceSize>(lightTileCount) *
                sizeof(GpuLightTile),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame.lightTileBuffer, frame.lightTileMemory);
        createBuffer(
            sizeof(GpuOccupancyHierarchy),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            frame.occupancyBuffer, frame.occupancyMemory);

        const VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .extent = {
                textureWidth,
                textureHeight,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        check(vkCreateImage(device_, &imageInfo, nullptr, &frame.textureImage),
              "vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, frame.textureImage, &requirements);
        const VkMemoryAllocateInfo allocationInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = requirements.size,
            .memoryTypeIndex = findMemoryType(
                requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        check(vkAllocateMemory(device_, &allocationInfo, nullptr,
                               &frame.textureMemory),
              "vkAllocateMemory");
        check(vkBindImageMemory(device_, frame.textureImage, frame.textureMemory, 0),
              "vkBindImageMemory");

        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = frame.textureImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        check(vkCreateImageView(device_, &viewInfo, nullptr, &frame.textureView),
              "vkCreateImageView");

        const VkImageCreateInfo visibilityImageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .extent = {
                lightingWidth,
                lightingHeight,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        check(vkCreateImage(device_, &visibilityImageInfo, nullptr,
                            &frame.visibilityImage),
              "vkCreateImage");
        VkMemoryRequirements visibilityRequirements{};
        vkGetImageMemoryRequirements(
            device_, frame.visibilityImage, &visibilityRequirements);
        const VkMemoryAllocateInfo visibilityAllocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = visibilityRequirements.size,
            .memoryTypeIndex = findMemoryType(
                visibilityRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        check(vkAllocateMemory(device_, &visibilityAllocation, nullptr,
                               &frame.visibilityMemory),
              "vkAllocateMemory");
        check(vkBindImageMemory(device_, frame.visibilityImage,
                                frame.visibilityMemory, 0),
              "vkBindImageMemory");
        const VkImageViewCreateInfo visibilityViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = frame.visibilityImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        check(vkCreateImageView(device_, &visibilityViewInfo, nullptr,
                                &frame.visibilityView),
              "vkCreateImageView");

        const VkImageCreateInfo derivedImageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .extent = {
                textureWidth,
                textureHeight,
                1,
            },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        check(vkCreateImage(device_, &derivedImageInfo, nullptr,
                            &frame.derivedImage),
              "vkCreateImage");
        VkMemoryRequirements derivedRequirements{};
        vkGetImageMemoryRequirements(
            device_, frame.derivedImage, &derivedRequirements);
        const VkMemoryAllocateInfo derivedAllocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = derivedRequirements.size,
            .memoryTypeIndex = findMemoryType(
                derivedRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        check(vkAllocateMemory(device_, &derivedAllocation, nullptr,
                               &frame.derivedMemory),
              "vkAllocateMemory");
        check(vkBindImageMemory(device_, frame.derivedImage,
                                frame.derivedMemory, 0),
              "vkBindImageMemory");
        const VkImageViewCreateInfo derivedViewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = frame.derivedImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UINT,
            .components = {},
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        check(vkCreateImageView(device_, &derivedViewInfo, nullptr,
                                &frame.derivedView),
              "vkCreateImageView");

        check(vkCreateImage(device_, &derivedImageInfo, nullptr,
                            &frame.hazeScratchImage),
              "vkCreateImage");
        VkMemoryRequirements hazeScratchRequirements{};
        vkGetImageMemoryRequirements(
            device_, frame.hazeScratchImage, &hazeScratchRequirements);
        const VkMemoryAllocateInfo hazeScratchAllocation{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = hazeScratchRequirements.size,
            .memoryTypeIndex = findMemoryType(
                hazeScratchRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        };
        check(vkAllocateMemory(device_, &hazeScratchAllocation, nullptr,
                               &frame.hazeScratchMemory),
              "vkAllocateMemory");
        check(vkBindImageMemory(device_, frame.hazeScratchImage,
                                frame.hazeScratchMemory, 0),
              "vkBindImageMemory");
        VkImageViewCreateInfo hazeScratchViewInfo = derivedViewInfo;
        hazeScratchViewInfo.image = frame.hazeScratchImage;
        check(vkCreateImageView(device_, &hazeScratchViewInfo, nullptr,
                                &frame.hazeScratchView),
              "vkCreateImageView");

        const auto createGiImage =
            [&](VkImage& image, VkDeviceMemory& memory,
                VkImageView& view) {
                VkImageCreateInfo giImageInfo =
                    visibilityImageInfo;
                giImageInfo.format =
                    VK_FORMAT_R16G16B16A16_SFLOAT;
                check(vkCreateImage(device_, &giImageInfo, nullptr,
                                    &image),
                      "vkCreateImage");
                VkMemoryRequirements giRequirements{};
                vkGetImageMemoryRequirements(device_, image,
                                             &giRequirements);
                const VkMemoryAllocateInfo giAllocation{
                    .sType =
                        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .pNext = nullptr,
                    .allocationSize = giRequirements.size,
                    .memoryTypeIndex = findMemoryType(
                        giRequirements.memoryTypeBits,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
                };
                check(vkAllocateMemory(device_, &giAllocation, nullptr,
                                       &memory),
                      "vkAllocateMemory");
                check(vkBindImageMemory(device_, image, memory, 0),
                      "vkBindImageMemory");
                VkImageViewCreateInfo giViewInfo =
                    visibilityViewInfo;
                giViewInfo.image = image;
                giViewInfo.format =
                    VK_FORMAT_R16G16B16A16_SFLOAT;
                check(vkCreateImageView(device_, &giViewInfo, nullptr,
                                        &view),
                      "vkCreateImageView");
            };
        createGiImage(frame.giIntermediateImage,
                      frame.giIntermediateMemory,
                      frame.giIntermediateView);
        createGiImage(frame.giFinalImage, frame.giFinalMemory,
                      frame.giFinalView);

        const VkDescriptorSetAllocateInfo descriptorAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = textureDescriptorPool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &textureDescriptorLayout_,
        };
        check(vkAllocateDescriptorSets(device_, &descriptorAllocateInfo,
                                       &frame.textureDescriptor),
              "vkAllocateDescriptorSets");
        const VkDescriptorImageInfo sampledImage{
            .sampler = textureSampler_,
            .imageView = frame.textureView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo storageImage{
            .sampler = VK_NULL_HANDLE,
            .imageView = frame.visibilityView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo sampledVisibility{
            .sampler = textureSampler_,
            .imageView = frame.hazeScratchView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo storageDerived{
            .sampler = VK_NULL_HANDLE,
            .imageView = frame.derivedView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo sampledDerived{
            .sampler = textureSampler_,
            .imageView = frame.derivedView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo storageHazeScratch{
            .sampler = VK_NULL_HANDLE,
            .imageView = frame.hazeScratchView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo storageGiIntermediate{
            .sampler = VK_NULL_HANDLE,
            .imageView = frame.giIntermediateView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo storageGiFinal{
            .sampler = VK_NULL_HANDLE,
            .imageView = frame.giFinalView,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorImageInfo sampledGiFinal{
            .sampler = textureSampler_,
            .imageView = frame.giFinalView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo storageSunTransmittance{
            .sampler = VK_NULL_HANDLE,
            .imageView = sunTransmittanceView_,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        const VkDescriptorBufferInfo particleBufferInfo{
            .buffer = particleBuffer_,
            .offset = 0,
            .range = particleBytes,
        };
        const VkDescriptorBufferInfo spawnBufferInfo{
            .buffer = frame.particleSpawnBuffer,
            .offset = 0,
            .range = particleBytes,
        };
        const VkDescriptorBufferInfo materialStateAInfo{
            .buffer = materialStateA_,
            .offset = 0,
            .range = materialBytes,
        };
        const VkDescriptorBufferInfo materialStateBInfo{
            .buffer = materialStateB_,
            .offset = 0,
            .range = materialBytes,
        };
        const VkDescriptorBufferInfo materialFluxInfo{
            .buffer = materialFluxBuffer_,
            .offset = 0,
            .range = materialFluxBytes,
        };
        const VkDescriptorBufferInfo sceneLightInfo{
            .buffer = frame.sceneLightBuffer,
            .offset = 0,
            .range = sizeof(GpuSceneLights),
        };
        const VkDescriptorBufferInfo lightTileInfo{
            .buffer = frame.lightTileBuffer,
            .offset = 0,
            .range = static_cast<VkDeviceSize>(lightTileCount) *
                     sizeof(GpuLightTile),
        };
        const VkDescriptorBufferInfo occupancyInfo{
            .buffer = frame.occupancyBuffer,
            .offset = 0,
            .range = sizeof(GpuOccupancyHierarchy),
        };
        const std::array descriptorWrites{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &sampledImage,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageImage,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &sampledVisibility,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageDerived,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 4,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &sampledDerived,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 5,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &particleBufferInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 6,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &spawnBufferInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 7,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &materialStateAInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 8,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &materialStateBInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 9,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &materialFluxInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 12,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &sceneLightInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 10,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageHazeScratch,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 13,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageGiIntermediate,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 14,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageGiFinal,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 15,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType =
                    VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &sampledGiFinal,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 16,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &lightTileInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 17,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &occupancyInfo,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = frame.textureDescriptor,
                .dstBinding = 18,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &storageSunTransmittance,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        };
        vkUpdateDescriptorSets(
            device_, static_cast<std::uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0, nullptr);
    }

    // Each frame samples the immediately preceding frame's filtered lighting
    // while writing only to its own image. Keeping those images separate
    // avoids read/write races when camera reprojection shifts history cells.
    for (std::size_t frameIndex = 0; frameIndex < framesInFlight;
         ++frameIndex) {
        FrameResources& frame = frames_[frameIndex];
        const FrameResources& historyFrame =
            frames_[(frameIndex + framesInFlight - 1) %
                    framesInFlight];
        const VkDescriptorImageInfo sampledHistory{
            .sampler = textureSampler_,
            .imageView = historyFrame.giFinalView,
            .imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkWriteDescriptorSet historyWrite{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = frame.textureDescriptor,
            .dstBinding = 11,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &sampledHistory,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device_, 1, &historyWrite, 0, nullptr);
    }
}

void VulkanRenderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties,
                                  VkBuffer& buffer, VkDeviceMemory& memory) {
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };
    check(vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer),
          "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, buffer, &requirements);
    const VkMemoryAllocateInfo allocationInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex =
            findMemoryType(requirements.memoryTypeBits, properties),
    };
    check(vkAllocateMemory(device_, &allocationInfo, nullptr, &memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(device_, buffer, memory, 0), "vkBindBufferMemory");
}

void VulkanRenderer::uploadParticleSpawns(FrameResources& frame, World& world) {
    std::vector<Particle> spawns = world.takeParticleSpawns();
    frame.particleSpawnCount = static_cast<std::uint32_t>(
        std::min<std::size_t>(spawns.size(), maximumGpuParticles));
    if (frame.particleSpawnCount == 0) {
        return;
    }

    const std::size_t first =
        spawns.size() - static_cast<std::size_t>(frame.particleSpawnCount);
    const VkDeviceSize byteCount =
        static_cast<VkDeviceSize>(frame.particleSpawnCount) *
        sizeof(GpuParticle);
    void* mapped = nullptr;
    check(vkMapMemory(device_, frame.particleSpawnMemory, 0, byteCount, 0,
                      &mapped),
          "vkMapMemory");
    auto* gpuSpawns = static_cast<GpuParticle*>(mapped);
    for (std::uint32_t index = 0; index < frame.particleSpawnCount; ++index) {
        const Particle& particle =
            spawns[first + static_cast<std::size_t>(index)];
        gpuSpawns[index] = GpuParticle{
            .position = {particle.position.x, particle.position.y},
            .velocity = {particle.velocity.x, particle.velocity.y},
            .colorAndLife = {
                particle.color[0],
                particle.color[1],
                particle.color[2],
                particle.lifetime,
            },
            .maximumLifeAndSize = {
                particle.maximumLifetime,
                particle.size,
            },
            .slot = particleSpawnCursor_,
            .padding = 0,
        };
        particleSpawnCursor_ =
            (particleSpawnCursor_ + 1U) % maximumGpuParticles;
    }
    vkUnmapMemory(device_, frame.particleSpawnMemory);
}

void VulkanRenderer::prepareMaterialSimulation(World& world) {
    // Pressure waves are a visual/gameplay augmentation of the latest
    // authoritative material state. Never run a compute catch-up spiral:
    // collapse queued ticks into one current-state pressure update.
    materialSimulationSteps_ =
        world.pendingGpuMaterialSteps_ > 0U ? 1U : 0U;
    world.pendingGpuMaterialSteps_ = 0U;
    if (materialSimulationSteps_ == 0) {
        return;
    }
    if (materialSimulationPending_) {
        throw std::logic_error(
            "Material simulation was submitted before its result was read");
    }

    const World::ActiveBounds bounds = world.activeBounds();
    materialSimulationOriginX_ = bounds.minX;
    materialSimulationOriginY_ = bounds.minY;
    materialSimulationWidth_ =
        static_cast<std::uint32_t>(bounds.maxX - bounds.minX);
    materialSimulationHeight_ =
        static_cast<std::uint32_t>(bounds.maxY - bounds.minY);
    if (materialSimulationWidth_ > maximumSimulationWidth ||
        materialSimulationHeight_ > maximumSimulationHeight) {
        throw std::runtime_error(
            "The active material region exceeds the GPU simulation buffer");
    }

    const VkDeviceSize cellCount =
        static_cast<VkDeviceSize>(materialSimulationWidth_) *
        materialSimulationHeight_;
    const VkDeviceSize byteCount = cellCount * sizeof(GpuMaterialCell);
    void* mapped = nullptr;
    check(vkMapMemory(device_, materialUploadMemory_, 0, byteCount, 0,
                      &mapped),
          "vkMapMemory");
    auto* gpuCells = static_cast<GpuMaterialCell*>(mapped);
    for (std::uint32_t localY = 0; localY < materialSimulationHeight_;
         ++localY) {
        const int worldY =
            materialSimulationOriginY_ + static_cast<int>(localY);
        for (std::uint32_t localX = 0; localX < materialSimulationWidth_;
             ++localX) {
            const int worldX =
                materialSimulationOriginX_ + static_cast<int>(localX);
            const std::size_t worldIndex = static_cast<std::size_t>(
                worldY * World::width + worldX);
            const std::size_t localIndex = static_cast<std::size_t>(
                localY * materialSimulationWidth_ + localX);
            gpuCells[localIndex] = GpuMaterialCell{
                .material = static_cast<std::uint32_t>(
                    world.cells_[worldIndex]),
                .liquidAmount = world.liquidAmount_[worldIndex],
                .flowX = world.liquidFlowX_[worldIndex],
                .flowY = world.liquidFlowY_[worldIndex],
                .heat = world.heat_[worldIndex],
                .burnProgress = world.burnProgress_[worldIndex],
                .gasLifetime = world.gasLifetime_[worldIndex],
                .gasDrift = world.gasDrift_[worldIndex],
                // The Noita-style cellular solver has no compressible
                // density/energy field. Keep neutral legacy values only so
                // old GPU material jobs remain binary-compatible while that
                // optional path is retired.
                .density = 1.0F,
                .momentumX = 0.0F,
                .momentumY = 0.0F,
                .totalEnergy = 2.5F,
            };
        }
    }
    vkUnmapMemory(device_, materialUploadMemory_);
    ++materialSimulationSeed_;
}

void VulkanRenderer::synchronizeMaterialSimulation(World& world) {
    if (!materialSimulationPending_) {
        return;
    }
    FrameResources& frame = frames_[materialSimulationFrame_];
    check(vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");

    const VkDeviceSize cellCount =
        static_cast<VkDeviceSize>(materialSimulationWidth_) *
        materialSimulationHeight_;
    const VkDeviceSize byteCount = cellCount * sizeof(GpuMaterialCell);
    void* mapped = nullptr;
    check(vkMapMemory(device_, materialReadbackMemory_, 0, byteCount, 0,
                      &mapped),
          "vkMapMemory");
    const auto* gpuCells = static_cast<const GpuMaterialCell*>(mapped);
    for (std::uint32_t localY = 0; localY < materialSimulationHeight_;
         ++localY) {
        const int worldY =
            materialSimulationOriginY_ + static_cast<int>(localY);
        for (std::uint32_t localX = 0; localX < materialSimulationWidth_;
             ++localX) {
            const int worldX =
                materialSimulationOriginX_ + static_cast<int>(localX);
            const std::size_t worldIndex = static_cast<std::size_t>(
                worldY * World::width + worldX);
            const std::size_t localIndex = static_cast<std::size_t>(
                localY * materialSimulationWidth_ + localX);
            const GpuMaterialCell& source = gpuCells[localIndex];
            // CPU cellular transport may have advanced after this legacy GPU
            // job was submitted. Read back only its directional hint; never
            // overwrite newer material identity or reaction state.
            world.liquidFlowX_[worldIndex] =
                static_cast<std::int8_t>(
                    std::clamp(source.flowX, -127, 127));
            world.liquidFlowY_[worldIndex] =
                static_cast<std::int8_t>(
                    std::clamp(source.flowY, -127, 127));
        }
    }
    vkUnmapMemory(device_, materialReadbackMemory_);
    materialSimulationPending_ = false;
}

std::uint32_t VulkanRenderer::findMemoryType(
    std::uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);
    for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
        if ((typeFilter & (1U << index)) != 0U &&
            (memoryProperties.memoryTypes[index].propertyFlags & properties) ==
                properties) {
            return index;
        }
    }
    throw std::runtime_error("No suitable Vulkan memory type was found");
}

std::vector<Vertex> VulkanRenderer::buildVertices(const World& world) const {
    std::vector<Vertex> vertices;
    vertices.reserve(8'192);

    const Vec2 camera = world.renderCameraTopLeft();
    const auto xToClip = [camera](float x) {
        return (x - camera.x) / static_cast<float>(World::viewWidth) * 2.0F -
               1.0F;
    };
    const auto yToClip = [camera](float y) {
        return (y - camera.y) / static_cast<float>(World::viewHeight) * 2.0F -
               1.0F;
    };
    const auto addWorldQuad = [&](float x, float y, float halfWidth,
                                  float halfHeight,
                                  std::array<float, 3> color,
                                  float alpha = 1.0F,
                                  float lightingStrength = 1.0F) {
        const float left = x - halfWidth;
        const float top = y - halfHeight;
        const float right = x + halfWidth;
        const float bottom = y + halfHeight;
        const auto vertex = [&](float worldX, float worldY) {
            return Vertex{
                xToClip(worldX), yToClip(worldY),
                color[0], color[1], color[2], alpha,
                worldX, worldY, lightingStrength,
            };
        };
        const Vertex topLeft = vertex(left, top);
        const Vertex topRight = vertex(right, top);
        const Vertex bottomLeft = vertex(left, bottom);
        const Vertex bottomRight = vertex(right, bottom);
        vertices.insert(vertices.end(), {
            topLeft, bottomLeft, bottomRight,
            topLeft, bottomRight, topRight,
        });
    };
    const auto addWorldLine = [&](Vec2 first, Vec2 second, float thickness,
                                  std::array<float, 3> color,
                                  float alpha = 1.0F,
                                  float lightingStrength = 1.0F) {
        const Vec2 direction = normalized(second - first);
        const Vec2 perpendicular{-direction.y * thickness,
                                 direction.x * thickness};
        const std::array<Vec2, 4> corners{
            first + perpendicular,
            first - perpendicular,
            second - perpendicular,
            second + perpendicular,
        };
        const auto vertex = [&](Vec2 point) {
            return Vertex{xToClip(point.x), yToClip(point.y),
                          color[0], color[1], color[2], alpha,
                          point.x, point.y, lightingStrength};
        };
        vertices.insert(vertices.end(), {
            vertex(corners[0]), vertex(corners[1]), vertex(corners[2]),
            vertex(corners[0]), vertex(corners[2]), vertex(corners[3]),
        });
    };
    const Grapple& grapple = world.grapple();
    if (grapple.active) {
        if (grapple.attached && grapple.points.size() >= 2) {
            for (std::size_t index = 1; index < grapple.points.size(); ++index) {
                addWorldLine(grapple.points[index - 1], grapple.points[index],
                             0.42F * renderScale,
                             {0.58F, 0.61F, 0.66F},
                             1.0F, 1.025F);
            }
        } else {
            addWorldLine(world.player().position, grapple.hookPosition,
                         0.32F * renderScale,
                         {0.50F, 0.53F, 0.58F}, 1.0F, 1.025F);
        }
        addWorldQuad(grapple.hookPosition.x, grapple.hookPosition.y,
                     1.15F * renderScale, 1.15F * renderScale,
                     {0.95F, 0.70F, 0.22F}, 1.0F, 1.035F);
    }

    const Player& player = world.player();
    constexpr float playerSpriteWidth = 6.0F * renderScale;
    constexpr float playerSpriteHeight = 12.0F * renderScale;
    constexpr float playerCollisionHalfHeight = 4.2F * renderScale;
    const float pixelWidth =
        playerSpriteWidth / static_cast<float>(playerSpriteWidth_);
    const float pixelHeight =
        playerSpriteHeight / static_cast<float>(playerSpriteHeight_);
    const float spriteLeft =
        player.position.x - playerSpriteWidth * 0.5F;
    const float spriteTop =
        player.position.y + playerCollisionHalfHeight -
        playerSpriteHeight;
    for (const SpritePixel& pixel : playerSprite_) {
        const bool emissiveVisor =
            pixel.color[1] > 0.42F &&
            pixel.color[1] > pixel.color[0] * 1.18F &&
            pixel.color[1] > pixel.color[2] * 1.12F;
        const float lightingStrength =
            emissiveVisor ? 1.32F : 1.035F;
        const float sourceX =
            player.facing >= 0.0F
                ? static_cast<float>(pixel.x) + 0.5F
                : static_cast<float>(
                      playerSpriteWidth_ - 1 -
                      static_cast<int>(pixel.x)) +
                      0.5F;
        const float centerX = spriteLeft + sourceX * pixelWidth;
        const float centerY =
            spriteTop +
            (static_cast<float>(pixel.y) + 0.5F) * pixelHeight;
        addWorldQuad(centerX, centerY, pixelWidth * 0.51F,
                     pixelHeight * 0.51F, pixel.color, 1.0F,
                     lightingStrength);
    }
    for (const Projectile& bullet : world.bullets()) {
        addWorldQuad(bullet.position.x, bullet.position.y,
                     0.7F * renderScale, 0.7F * renderScale,
                     {1.0F, 0.91F, 0.45F}, 1.0F, 0.0F);
    }
    for (const Projectile& grenade : world.grenades()) {
        addWorldQuad(grenade.position.x, grenade.position.y,
                     1.5F * renderScale, 1.5F * renderScale,
                     {0.35F, 0.86F, 0.30F}, 1.0F, 0.0F);
    }
    for (const Particle& particle : world.particles()) {
        const float alpha = particle.maximumLifetime > 0.0F
                                ? std::clamp(particle.lifetime /
                                                 particle.maximumLifetime,
                                             0.0F, 1.0F)
                                : 0.0F;
        addWorldQuad(particle.position.x, particle.position.y, particle.size,
                     particle.size, particle.color, alpha, 0.0F);
    }

    constexpr std::array palette{
        Material::sand,
        Material::water,
        Material::oil,
        Material::fire,
        Material::smoke,
        Material::wood,
        Material::metal,
    };
    for (std::size_t index = 0; index < palette.size(); ++index) {
        const Material material = palette[index];
        const float x = camera.x +
                        (7.0F + static_cast<float>(index) * 8.0F) *
                            renderScale;
        const float y = camera.y + 7.0F * renderScale;
        const std::array<float, 3> border =
            material == world.selectedMaterial()
                ? std::array<float, 3>{0.96F, 0.98F, 1.0F}
                : std::array<float, 3>{0.10F, 0.12F, 0.16F};
        addWorldQuad(x, y, 3.5F * renderScale, 3.5F * renderScale,
                     border, 1.0F, 0.0F);
        addWorldQuad(x, y, 2.7F * renderScale, 2.7F * renderScale,
                     materialColor(material, static_cast<int>(index), 0),
                     1.0F, 0.0F);
    }

    const auto addHudBar = [&](float screenX, float screenY, float barWidth,
                               float barHeight, float value,
                               std::array<float, 3> color) {
        screenX *= renderScale;
        screenY *= renderScale;
        barWidth *= renderScale;
        barHeight *= renderScale;
        const float clamped = std::clamp(value, 0.0F, 1.0F);
        addWorldQuad(camera.x + screenX + barWidth * 0.5F,
                     camera.y + screenY + barHeight * 0.5F,
                     barWidth * 0.5F + 0.8F * renderScale,
                     barHeight * 0.5F + 0.8F * renderScale,
                     {0.08F, 0.09F, 0.12F}, 0.92F, 0.0F);
        if (clamped > 0.0F) {
            const float filledWidth = barWidth * clamped;
            addWorldQuad(camera.x + screenX + filledWidth * 0.5F,
                         camera.y + screenY + barHeight * 0.5F,
                         filledWidth * 0.5F, barHeight * 0.5F, color,
                         0.94F, 0.0F);
        }
    };
    const Player& hudPlayer = world.player();
    constexpr float hudLeft =
        320.0F - 46.0F;
    addHudBar(hudLeft, 5.0F, 40.0F, 3.0F, hudPlayer.health / 100.0F,
              {0.88F, 0.16F, 0.13F});
    addHudBar(hudLeft, 11.0F, 8.0F, 2.0F, hudPlayer.wetness,
              {0.08F, 0.48F, 0.96F});
    addHudBar(hudLeft + 10.5F, 11.0F, 8.0F, 2.0F, hudPlayer.oiliness,
              {0.42F, 0.18F, 0.48F});
    addHudBar(hudLeft + 21.0F, 11.0F, 8.0F, 2.0F, hudPlayer.burning,
              {1.0F, 0.25F, 0.02F});
    addHudBar(hudLeft + 31.5F, 11.0F, 8.0F, 2.0F, hudPlayer.suffocation,
              {0.50F, 0.53F, 0.58F});

    if (world.explosionFlash() > 0.0F) {
        appendQuad(vertices, -1.0F, -1.0F, 1.0F, 1.0F,
                   {1.0F, 0.70F, 0.34F}, world.explosionFlash());
    }
    return vertices;
}

void VulkanRenderer::updateSceneLights(const World& world, Vec2 camera) {
    playerLight_ = {
        world.player().position.x,
        world.player().position.y,
        rayTracingSettings_.playerLightRadius,
        playerLightEnabled_
            ? rayTracingSettings_.playerLightIntensity
            : 0.0F,
    };

    const std::uint64_t ticks = SDL_GetTicks64();
    const float smoothingDeltaTime =
        lastFireLightSmoothingTicks_ == 0
            ? 1.0F / 60.0F
            : std::clamp(
                  static_cast<float>(
                      ticks - lastFireLightSmoothingTicks_) *
                      0.001F,
                  0.0F, 0.1F);
    lastFireLightSmoothingTicks_ = ticks;

    // Fire cells evolve on the 30 Hz material simulation, so cluster
    // detection runs at that rate. The persistent light tracks below are
    // smoothed every rendered frame.
    if (lastFireLightUpdateTicks_ != 0 &&
        ticks - lastFireLightUpdateTicks_ < 33) {
        const float positionBlend =
            1.0F - std::exp(-smoothingDeltaTime / 0.12F);
        const float radiusBlend =
            1.0F - std::exp(-smoothingDeltaTime / 0.16F);
        for (std::size_t index = 0; index < fireLights_.size();
             ++index) {
            auto& light = fireLights_[index];
            const auto& target = fireLightTargets_[index];
            light[0] += (target[0] - light[0]) * positionBlend;
            light[1] += (target[1] - light[1]) * positionBlend;
            light[2] += (target[2] - light[2]) * radiusBlend;
            const float intensityTime =
                target[3] > light[3] ? 0.12F : 0.38F;
            const float intensityBlend =
                1.0F -
                std::exp(-smoothingDeltaTime / intensityTime);
            light[3] +=
                (target[3] - light[3]) * intensityBlend;
            if (light[3] < 0.001F && target[3] <= 0.0F) {
                light = {};
            }
        }
        return;
    }
    lastFireLightUpdateTicks_ = ticks;

    const int visibleLeft = std::clamp(
        static_cast<int>(std::floor(camera.x)), 0, World::width - 1);
    const int visibleTop = std::clamp(
        static_cast<int>(std::floor(camera.y)), 0, World::height - 1);
    const int visibleRight =
        std::min(World::width, visibleLeft + World::viewWidth + 2);
    const int visibleBottom =
        std::min(World::height, visibleTop + World::viewHeight + 2);

    struct EmissiveCluster {
        float weight = 0.0F;
        float weightedX = 0.0F;
        float weightedY = 0.0F;
    };
    const int regionWidth = visibleRight - visibleLeft;
    const int regionHeight = visibleBottom - visibleTop;
    const auto emissionAt = [&](int localX, int localY) {
        const int worldX = visibleLeft + localX;
        const int worldY = visibleTop + localY;
        const std::size_t index =
            static_cast<std::size_t>(worldY * World::width + worldX);
        const Material material = world.materials()[index];
        if (material == Material::fire) {
            return 1.0F;
        }
        if (material == Material::wood &&
            world.heat()[index] > 0.52F) {
            return world.heat()[index] * 0.55F;
        }
        return 0.0F;
    };

    // Find actual connected emitters first. The old quadrant average could
    // place a light halfway between unrelated fires, leaving the flames
    // themselves dark and illuminating empty space.
    std::vector<std::uint8_t> visited(
        static_cast<std::size_t>(regionWidth * regionHeight), 0);
    std::vector<int> frontier;
    std::vector<EmissiveCluster> components;
    frontier.reserve(256);
    components.reserve(32);
    for (int localY = 0; localY < regionHeight; ++localY) {
        for (int localX = 0; localX < regionWidth; ++localX) {
            const int localIndex = localY * regionWidth + localX;
            if (visited[static_cast<std::size_t>(localIndex)] != 0 ||
                emissionAt(localX, localY) <= 0.0F) {
                continue;
            }

            EmissiveCluster component;
            frontier.clear();
            frontier.push_back(localIndex);
            visited[static_cast<std::size_t>(localIndex)] = 1;
            for (std::size_t cursor = 0; cursor < frontier.size();
                 ++cursor) {
                const int current = frontier[cursor];
                const int currentX = current % regionWidth;
                const int currentY = current / regionWidth;
                const float weight = emissionAt(currentX, currentY);
                component.weight += weight;
                component.weightedX +=
                    (static_cast<float>(visibleLeft + currentX) + 0.5F) *
                    weight;
                component.weightedY +=
                    (static_cast<float>(visibleTop + currentY) + 0.5F) *
                    weight;

                for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                        if (offsetX == 0 && offsetY == 0) {
                            continue;
                        }
                        const int neighborX = currentX + offsetX;
                        const int neighborY = currentY + offsetY;
                        if (neighborX < 0 || neighborX >= regionWidth ||
                            neighborY < 0 ||
                            neighborY >= regionHeight) {
                            continue;
                        }
                        const int neighbor =
                            neighborY * regionWidth + neighborX;
                        if (visited[static_cast<std::size_t>(neighbor)] !=
                                0 ||
                            emissionAt(neighborX, neighborY) <= 0.0F) {
                            continue;
                        }
                        visited[static_cast<std::size_t>(neighbor)] = 1;
                        frontier.push_back(neighbor);
                    }
                }
            }
            components.push_back(component);
        }
    }

    std::ranges::sort(
        components, std::greater{},
        [](const EmissiveCluster& cluster) { return cluster.weight; });
    std::vector<EmissiveCluster> sources;
    sources.reserve(components.size());
    const float mergeDistance =
        10.0F * static_cast<float>(World::simulationScale);
    const float mergeDistanceSquared = mergeDistance * mergeDistance;
    for (const EmissiveCluster& component : components) {
        const float componentX =
            component.weightedX / component.weight;
        const float componentY =
            component.weightedY / component.weight;
        EmissiveCluster* nearest = nullptr;
        float nearestDistanceSquared = mergeDistanceSquared;
        for (EmissiveCluster& source : sources) {
            const float sourceX = source.weightedX / source.weight;
            const float sourceY = source.weightedY / source.weight;
            const float deltaX = sourceX - componentX;
            const float deltaY = sourceY - componentY;
            const float distanceSquared =
                deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared < nearestDistanceSquared) {
                nearestDistanceSquared = distanceSquared;
                nearest = &source;
            }
        }
        if (nearest != nullptr) {
            nearest->weight += component.weight;
            nearest->weightedX += component.weightedX;
            nearest->weightedY += component.weightedY;
        } else {
            sources.push_back(component);
        }
    }
    std::ranges::sort(
        sources, std::greater{},
        [](const EmissiveCluster& cluster) { return cluster.weight; });

    std::vector<std::array<float, 4>> candidates;
    candidates.reserve(sources.size());
    for (const EmissiveCluster& source : sources) {
        candidates.push_back({
            source.weightedX / source.weight,
            source.weightedY / source.weight,
            rayTracingSettings_.fireBaseRadius +
                std::min(std::sqrt(source.weight) *
                             rayTracingSettings_.fireClusterRadiusScale,
                         rayTracingSettings_.fireMaximumRadiusBonus),
            std::min(rayTracingSettings_.fireMaximumIntensity,
                     rayTracingSettings_.fireBaseIntensity +
                         source.weight *
                             rayTracingSettings_.fireClusterIntensityScale),
        });
    }
    if (candidates.size() > maximumFireLights) {
        candidates.resize(maximumFireLights);
    }

    struct LightMatch {
        float distanceSquared;
        std::size_t slot;
        std::size_t source;
    };
    std::vector<LightMatch> possibleMatches;
    const float trackingDistance =
        18.0F * static_cast<float>(World::simulationScale);
    const float trackingDistanceSquared =
        trackingDistance * trackingDistance;
    for (std::size_t slot = 0; slot < fireLights_.size(); ++slot) {
        const bool active =
            fireLights_[slot][3] > 0.02F ||
            fireLightTargets_[slot][3] > 0.0F;
        if (!active) {
            continue;
        }
        const auto& reference =
            fireLightTargets_[slot][3] > 0.0F
                ? fireLightTargets_[slot]
                : fireLights_[slot];
        for (std::size_t source = 0; source < candidates.size();
             ++source) {
            const float deltaX =
                reference[0] - candidates[source][0];
            const float deltaY =
                reference[1] - candidates[source][1];
            const float distanceSquared =
                deltaX * deltaX + deltaY * deltaY;
            if (distanceSquared <= trackingDistanceSquared) {
                possibleMatches.push_back(
                    {distanceSquared, slot, source});
            }
        }
    }
    std::ranges::sort(
        possibleMatches, std::less{},
        [](const LightMatch& match) {
            return match.distanceSquared;
        });

    std::array<std::array<float, 4>, maximumFireLights> newTargets{};
    for (std::size_t slot = 0; slot < fireLights_.size(); ++slot) {
        // An emitter that disappears should fade where it was last seen.
        // Leaving an unmatched target at all-zero values makes the light
        // interpolate toward world origin while its intensity decays.
        newTargets[slot] = {
            fireLights_[slot][0],
            fireLights_[slot][1],
            fireLights_[slot][2],
            0.0F,
        };
    }
    std::array<bool, maximumFireLights> assignedSlots{};
    std::vector<bool> assignedSources(candidates.size(), false);
    for (const LightMatch& match : possibleMatches) {
        if (assignedSlots[match.slot] ||
            assignedSources[match.source]) {
            continue;
        }
        newTargets[match.slot] = candidates[match.source];
        assignedSlots[match.slot] = true;
        assignedSources[match.source] = true;
    }

    // Preserve established lights first. Only genuinely inactive slots are
    // used for new fires, preventing the fifth-strongest cluster from
    // repeatedly displacing the fourth as their cell counts fluctuate.
    for (std::size_t source = 0; source < candidates.size();
         ++source) {
        if (assignedSources[source]) {
            continue;
        }
        for (std::size_t slot = 0; slot < fireLights_.size();
             ++slot) {
            if (assignedSlots[slot] ||
                fireLights_[slot][3] >= 0.02F) {
                continue;
            }
            newTargets[slot] = candidates[source];
            assignedSlots[slot] = true;
            assignedSources[source] = true;
            fireLights_[slot][0] = candidates[source][0];
            fireLights_[slot][1] = candidates[source][1];
            fireLights_[slot][2] = candidates[source][2];
            fireLights_[slot][3] = 0.0F;
            break;
        }
    }
    fireLightTargets_ = newTargets;

    const float positionBlend =
        1.0F - std::exp(-smoothingDeltaTime / 0.12F);
    const float radiusBlend =
        1.0F - std::exp(-smoothingDeltaTime / 0.16F);
    for (std::size_t index = 0; index < fireLights_.size();
         ++index) {
        auto& light = fireLights_[index];
        const auto& target = fireLightTargets_[index];
        light[0] += (target[0] - light[0]) * positionBlend;
        light[1] += (target[1] - light[1]) * positionBlend;
        light[2] += (target[2] - light[2]) * radiusBlend;
        const float intensityTime =
            target[3] > light[3] ? 0.12F : 0.38F;
        const float intensityBlend =
            1.0F - std::exp(-smoothingDeltaTime / intensityTime);
        light[3] += (target[3] - light[3]) * intensityBlend;
        if (light[3] < 0.001F && target[3] <= 0.0F) {
            light = {};
        }
    }
}

void VulkanRenderer::recordCommands(VkCommandBuffer commandBuffer,
                                    std::uint32_t imageIndex,
                                    std::uint32_t vertexCount,
                                    const World& world) {
    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0,
        .pInheritanceInfo = nullptr,
    };
    check(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    FrameResources& frame = frames_[currentFrame_];
    vkCmdResetQueryPool(commandBuffer, frame.rayTimingQueryPool, 0,
                        rayTimingQueryCount);
    const Vec2 camera = world.renderCameraTopLeft();
    const bool temporalSettingsChanged =
        !temporalHistorySettingsInitialized_ ||
        temporalHistorySettings_.indirectBounces !=
            rayTracingSettings_.indirectBounces ||
        temporalHistorySettings_.giRays !=
            rayTracingSettings_.giRays ||
        temporalHistorySettings_.giIntensity !=
            rayTracingSettings_.giIntensity ||
        temporalHistorySettings_.bounceDistance !=
            rayTracingSettings_.bounceDistance ||
        temporalHistorySettings_.temporalGiDenoising !=
            rayTracingSettings_.temporalGiDenoising ||
        temporalHistorySettings_.giHistoryWeight !=
            rayTracingSettings_.giHistoryWeight ||
        temporalHistorySettings_.adaptiveGiSampling !=
            rayTracingSettings_.adaptiveGiSampling;
    if (temporalSettingsChanged) {
        for (FrameResources& resource : frames_) {
            resource.lightingHistoryValid = false;
        }
        temporalHistorySettings_ = rayTracingSettings_;
        temporalHistorySettingsInitialized_ = true;
    }
    FrameResources& historyFrame =
        frames_[(currentFrame_ + framesInFlight - 1) %
                framesInFlight];
    const bool historyImageInitialized =
        historyFrame.lightingHistoryValid;
    const bool temporalHistoryValid =
        rayTracingSettings_.temporalGiDenoising &&
        historyImageInitialized;
    const std::int32_t lightingOriginX =
        static_cast<std::int32_t>(std::floor(camera.x));
    const std::int32_t lightingOriginY =
        static_cast<std::int32_t>(std::floor(camera.y));
    const std::int32_t historyOffsetX = std::clamp(
        lightingOriginX - historyFrame.lightingOriginX,
        -32768, 32767);
    const std::int32_t historyOffsetY = std::clamp(
        lightingOriginY - historyFrame.lightingOriginY,
        -32768, 32767);
    const std::uint32_t packedHistoryOffset =
        static_cast<std::uint32_t>(
            static_cast<std::uint16_t>(historyOffsetX)) |
        (static_cast<std::uint32_t>(
             static_cast<std::uint16_t>(historyOffsetY))
         << 16U);
    if (!particleBufferInitialized_) {
        vkCmdFillBuffer(commandBuffer, particleBuffer_, 0, VK_WHOLE_SIZE, 0);
        const VkBufferMemoryBarrier initializedParticles{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = particleBuffer_,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &initializedParticles, 0, nullptr);
        particleBufferInitialized_ = true;
    } else {
        const VkBufferMemoryBarrier particlesToCompute{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = particleBuffer_,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1,
            &particlesToCompute, 0, nullptr);
    }

    struct ParticlePush {
        float deltaTime;
        std::uint32_t stage;
        std::uint32_t spawnCount;
        std::uint32_t capacity;
    };
    static_assert(sizeof(ParticlePush) == particlePushConstantBytes);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      particleComputePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        particleComputePipelineLayout_, 0, 1, &frame.textureDescriptor, 0,
        nullptr);
    ParticlePush particlePush{
        .deltaTime = particleDeltaTime_,
        .stage = 0,
        .spawnCount = frame.particleSpawnCount,
        .capacity = maximumGpuParticles,
    };
    vkCmdPushConstants(commandBuffer, particleComputePipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(particlePush), &particlePush);
    vkCmdDispatch(commandBuffer, (maximumGpuParticles + 255U) / 256U, 1, 1);

    VkBufferMemoryBarrier particleComputeBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = particleBuffer_,
        .offset = 0,
        .size = VK_WHOLE_SIZE,
    };
    if (frame.particleSpawnCount > 0) {
        vkCmdPipelineBarrier(commandBuffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 1, &particleComputeBarrier, 0, nullptr);
        particlePush.stage = 1;
        vkCmdPushConstants(commandBuffer, particleComputePipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(particlePush), &particlePush);
        vkCmdDispatch(commandBuffer,
                      (frame.particleSpawnCount + 255U) / 256U, 1, 1);
    }
    particleComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr,
                         1, &particleComputeBarrier, 0, nullptr);

    if (materialSimulationSteps_ > 0) {
        const VkDeviceSize materialByteCount =
            static_cast<VkDeviceSize>(materialSimulationWidth_) *
            materialSimulationHeight_ * sizeof(GpuMaterialCell);
        const VkBufferCopy materialUpload{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = materialByteCount,
        };
        vkCmdCopyBuffer(commandBuffer, materialUploadBuffer_,
                        materialStateA_, 1, &materialUpload);
        const std::array materialBuffers{
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = materialStateA_,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            },
            VkBufferMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer = materialStateB_,
                .offset = 0,
                .size = VK_WHOLE_SIZE,
            },
        };
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
            static_cast<std::uint32_t>(materialBuffers.size()),
            materialBuffers.data(), 0, nullptr);

        struct alignas(16) MaterialPush {
            std::array<std::int32_t, 4> region;
            std::array<std::uint32_t, 4> options;
        };
        static_assert(sizeof(MaterialPush) == materialPushConstantBytes);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          materialComputePipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            materialComputePipelineLayout_, 0, 1,
            &frame.textureDescriptor, 0, nullptr);
        std::uint32_t ping = 0;
        constexpr std::array<std::uint32_t, 4> pressureStages{
            19U, 20U, 21U, 22U};
        for (std::uint32_t step = 0; step < materialSimulationSteps_;
             ++step) {
            for (const std::uint32_t stage : pressureStages) {
                const MaterialPush materialPush{
                    .region = {
                        static_cast<std::int32_t>(
                            materialSimulationWidth_),
                        static_cast<std::int32_t>(
                            materialSimulationHeight_),
                        materialSimulationOriginX_,
                        materialSimulationOriginY_,
                    },
                    .options = {
                        stage,
                        ping,
                        materialSimulationSeed_ + step,
                        0,
                    },
                };
                vkCmdPushConstants(
                    commandBuffer, materialComputePipelineLayout_,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    sizeof(materialPush), &materialPush);
                vkCmdDispatch(
                    commandBuffer,
                    (materialSimulationWidth_ + 7U) / 8U,
                    (materialSimulationHeight_ + 7U) / 8U, 1);
                // The flux stage writes only the scratch field; the
                // authoritative state remains the input for liquid apply.
                const bool liquidFluxStage =
                    stage >= 1U && stage < 13U &&
                    (stage - 1U) % 3U == 1U;
                if (!liquidFluxStage) {
                    ping ^= 1U;
                }

                const std::array betweenStages{
                    VkBufferMemoryBarrier{
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask =
                            VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT,
                        .srcQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .buffer = materialStateA_,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                    },
                    VkBufferMemoryBarrier{
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask =
                            VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT,
                        .srcQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .buffer = materialStateB_,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                    },
                    VkBufferMemoryBarrier{
                        .sType =
                            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask =
                            VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT,
                        .srcQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex =
                            VK_QUEUE_FAMILY_IGNORED,
                        .buffer = materialFluxBuffer_,
                        .offset = 0,
                        .size = VK_WHOLE_SIZE,
                    },
                };
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                    nullptr,
                    static_cast<std::uint32_t>(
                        betweenStages.size()),
                    betweenStages.data(), 0, nullptr);
            }
        }
        materialSimulationResultInB_ = ping != 0;
        VkBufferMemoryBarrier toReadback{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = materialSimulationResultInB_
                          ? materialStateB_
                          : materialStateA_,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
        };
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
            &toReadback, 0,
            nullptr);
        const VkBufferCopy materialReadback{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = materialByteCount,
        };
        vkCmdCopyBuffer(
            commandBuffer,
            materialSimulationResultInB_ ? materialStateB_
                                         : materialStateA_,
            materialReadbackBuffer_, 1, &materialReadback);
        const VkBufferMemoryBarrier readbackToHost{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = materialReadbackBuffer_,
            .offset = 0,
            .size = materialByteCount,
        };
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
            &readbackToHost, 0, nullptr);
    }

    const VkImageMemoryBarrier toTransfer{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            frame.textureInitialized ? VK_ACCESS_SHADER_READ_BIT : 0U,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = frame.textureInitialized
                         ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                         : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.textureImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
        &toTransfer);
    const VkBufferImageCopy copyRegion{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {
            textureWidth,
            textureHeight,
            1,
        },
    };
    vkCmdCopyBufferToImage(commandBuffer, frame.textureStagingBuffer,
                           frame.textureImage,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
    const VkImageMemoryBarrier toCompute{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.textureImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &toCompute);

    const VkImageMemoryBarrier historyToCompute{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            historyImageInitialized
                ? VK_ACCESS_SHADER_WRITE_BIT |
                      VK_ACCESS_SHADER_READ_BIT
                : 0U,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout =
            historyImageInitialized
                ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = historyFrame.giFinalImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        historyImageInitialized
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &historyToCompute);

    const VkImageMemoryBarrier visibilityToCompute{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            frame.textureInitialized
                ? VK_ACCESS_SHADER_READ_BIT |
                      VK_ACCESS_SHADER_WRITE_BIT
                : 0U,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = frame.textureInitialized
                         ? VK_IMAGE_LAYOUT_GENERAL
                         : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.visibilityImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &visibilityToCompute);

    VkImageMemoryBarrier giIntermediateToCompute =
        visibilityToCompute;
    giIntermediateToCompute.image = frame.giIntermediateImage;
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &giIntermediateToCompute);

    VkImageMemoryBarrier giFinalToCompute = visibilityToCompute;
    giFinalToCompute.image = frame.giFinalImage;
    giFinalToCompute.oldLayout =
        frame.textureInitialized
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &giFinalToCompute);

    VkImageMemoryBarrier derivedToCompute = visibilityToCompute;
    derivedToCompute.image = frame.derivedImage;
    derivedToCompute.srcAccessMask =
        frame.textureInitialized ? VK_ACCESS_SHADER_READ_BIT : 0U;
    derivedToCompute.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    derivedToCompute.oldLayout =
        frame.textureInitialized
            ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED;
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                                 : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &derivedToCompute);

    const VkImageMemoryBarrier hazeScratchToCompute{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            frame.textureInitialized ? VK_ACCESS_SHADER_READ_BIT : 0U,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = frame.textureInitialized
                         ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                         : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.hazeScratchImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        frame.textureInitialized
            ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &hazeScratchToCompute);

    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        frame.rayTimingQueryPool, rayTimingStart);
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, derivedTimingStart);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      derivedPipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        derivedPipelineLayout_, 0, 1, &frame.textureDescriptor, 0,
        nullptr);
    std::uint32_t derivedStage = 0;
    vkCmdPushConstants(commandBuffer, derivedPipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(derivedStage), &derivedStage);
    vkCmdDispatch(commandBuffer, (textureWidth + 7U) / 8U,
                  (textureHeight + 7U) / 8U, 1);
    const VkImageMemoryBarrier hazeHorizontalToVertical{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.hazeScratchImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &hazeHorizontalToVertical);
    derivedStage = 1;
    vkCmdPushConstants(commandBuffer, derivedPipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(derivedStage), &derivedStage);
    vkCmdDispatch(commandBuffer, (textureWidth + 7U) / 8U,
                  (textureHeight + 7U) / 8U, 1);
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, derivedTimingEnd);

    const VkImageMemoryBarrier derivedFieldsToLighting{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.derivedImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &derivedFieldsToLighting);

    const VkImageMemoryBarrier hazeToShadowTrace{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.hazeScratchImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &hazeToShadowTrace);

    struct alignas(16) TracePush {
        std::array<std::int32_t, 4> originAndOptions;
        std::array<float, 4> transmission;
        std::array<float, 4> indirect;
        std::array<float, 4> celestial;
        std::array<std::int32_t, 4> sunCache;
    };
    static_assert(sizeof(TracePush) == tracePushConstantBytes);
    const std::uint32_t temporalFrameSeed =
        rayTracingSettings_.temporalGiDenoising
            ? (lightingFrameIndex_ & 0x3fffU)
            : 0U;
    const std::int32_t packedLightingOptions =
        (world.liquidDebug() &&
                 rayTracingSettings_.debugDisplayMode == 0
             ? 1
             : 0) |
        (std::clamp(rayTracingSettings_.indirectBounces, 0, 8) << 4) |
        (std::clamp(rayTracingSettings_.giRays, 1, 32) << 8) |
        (rayTracingSettings_.adaptiveGiSampling &&
                 rayTracingSettings_.temporalGiDenoising
             ? (1 << 3)
             : 0) |
        (temporalHistoryValid ? (1 << 16) : 0) |
        static_cast<std::int32_t>(temporalFrameSeed << 17U);
    const CelestialState celestial =
        celestialState(rayTracingSettings_);
    const std::int32_t packedDirectRayCounts =
        std::clamp(rayTracingSettings_.raysPerLight, 1, 1024) |
        (std::clamp(rayTracingSettings_.sunRays, 1, 16) << 16);
    const TracePush tracePush{
        .originAndOptions = {
            static_cast<std::int32_t>(std::floor(camera.x)),
            static_cast<std::int32_t>(std::floor(camera.y)),
            packedLightingOptions,
            packedDirectRayCounts,
        },
        .transmission = {
            rayTracingSettings_.softShadowRadius,
            rayTracingSettings_.solidTransmission,
            rayTracingSettings_.smokeTransmission,
            rayTracingSettings_.steamTransmission,
        },
        .indirect = {
            rayTracingSettings_.giIntensity,
            rayTracingSettings_.bounceDistance,
            rayTracingSettings_.giHistoryWeight,
            std::bit_cast<float>(packedHistoryOffset),
        },
        .celestial = {
            celestial.sunDirection.x,
            celestial.sunDirection.y,
            celestial.daylight,
            rayTracingSettings_.sunShadowSoftness,
        },
        .sunCache = {
            static_cast<std::int32_t>(std::floor(camera.x)) -
                sunTransmittanceOriginX_,
            static_cast<std::int32_t>(std::floor(camera.y)) -
                sunTransmittanceOriginY_,
            sunTransmittanceCacheValid_ ? 1 : 0,
            0,
        },
    };
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, directTimingStart);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      computePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        computePipelineLayout_, 0, 1, &frame.textureDescriptor, 0,
        nullptr);
    const VkImageMemoryBarrier sunTransmittanceToCompute{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            sunTransmittanceImageInitialized_
                ? VK_ACCESS_SHADER_READ_BIT |
                      VK_ACCESS_SHADER_WRITE_BIT
                : 0U,
        .dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout =
            sunTransmittanceImageInitialized_
                ? VK_IMAGE_LAYOUT_GENERAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = sunTransmittanceImage_,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer,
        sunTransmittanceImageInitialized_
            ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &sunTransmittanceToCompute);
    sunTransmittanceImageInitialized_ = true;
    if (updateSunTransmittanceThisFrame_) {
        TracePush cachePush = tracePush;
        cachePush.sunCache[0] = 0;
        cachePush.sunCache[1] = 0;
        cachePush.sunCache[2] = 0;
        cachePush.sunCache[3] = 1;
        vkCmdPushConstants(commandBuffer, computePipelineLayout_,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(cachePush), &cachePush);
        vkCmdDispatch(commandBuffer, (lightingWidth + 7U) / 8U,
                      (lightingHeight + 7U) / 8U, 1);
        const VkImageMemoryBarrier generatedSunCacheToDirect{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sunTransmittanceImage_,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        vkCmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
            nullptr, 1, &generatedSunCacheToDirect);
    }
    vkCmdPushConstants(commandBuffer, computePipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(tracePush), &tracePush);
    vkCmdDispatch(commandBuffer, (lightingWidth + 7U) / 8U,
                  (lightingHeight + 7U) / 8U, 1);
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, directTimingEnd);

    const VkImageMemoryBarrier directLightingCacheToGi{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.hazeScratchImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    VkImageMemoryBarrier directColorCacheToGi =
        directLightingCacheToGi;
    directColorCacheToGi.image = frame.giFinalImage;
    const std::array directCachesToGi{
        directLightingCacheToGi,
        directColorCacheToGi,
    };
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr,
        static_cast<std::uint32_t>(directCachesToGi.size()),
        directCachesToGi.data());

    TracePush giPush = tracePush;
    giPush.originAndOptions[2] |= 4;
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, giTimingStart);
    vkCmdPushConstants(commandBuffer, computePipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(giPush), &giPush);
    vkCmdDispatch(commandBuffer, (lightingWidth + 7U) / 8U,
                  (lightingHeight + 7U) / 8U, 1);
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, giTimingEnd);

    const std::array giToFilter{
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame.hazeScratchImage,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame.giIntermediateImage,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame.giFinalImage,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame.visibilityImage,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = frame.derivedImage,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        },
    };
    vkCmdPipelineBarrier(
        commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, static_cast<std::uint32_t>(giToFilter.size()),
        giToFilter.data());

    TracePush filterPush = tracePush;
    filterPush.originAndOptions[2] |= 2;
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, denoiseTimingStart);
    vkCmdPushConstants(commandBuffer, computePipelineLayout_,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(filterPush), &filterPush);
    vkCmdDispatch(commandBuffer, (lightingWidth + 7U) / 8U,
                  (lightingHeight + 7U) / 8U, 1);
    vkCmdWriteTimestamp(commandBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        frame.rayTimingQueryPool, denoiseTimingEnd);
    vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        frame.rayTimingQueryPool, rayTimingEnd);

    const VkImageMemoryBarrier rawLightingToFragment{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.visibilityImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1,
                         &rawLightingToFragment);
    VkImageMemoryBarrier rawGiToFragment =
        rawLightingToFragment;
    rawGiToFragment.image = frame.giIntermediateImage;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1,
                         &rawGiToFragment);

    const VkImageMemoryBarrier toFragment{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = frame.hazeScratchImage,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &toFragment);
    VkImageMemoryBarrier giToFragment = toFragment;
    giToFragment.image = frame.giFinalImage;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &giToFragment);
    VkImageMemoryBarrier derivedToFragment = toFragment;
    derivedToFragment.image = frame.derivedImage;
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1,
                         &derivedToFragment);
    frame.textureInitialized = true;

    const VkClearValue clearColor{{{0.025F, 0.035F, 0.055F, 1.0F}}};
    const VkRenderPassBeginInfo renderPassInfo{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = renderPass_,
        .framebuffer = framebuffers_[imageIndex],
        .renderArea = {{0, 0}, swapchainExtent_},
        .clearValueCount = 1,
        .pClearValues = &clearColor,
    };
    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo,
                         VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{
        .x = 0.0F,
        .y = 0.0F,
        .width = static_cast<float>(swapchainExtent_.width),
        .height = static_cast<float>(swapchainExtent_.height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };
    const VkRect2D scissor{{0, 0}, swapchainExtent_};
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      cellPipeline_);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            cellPipelineLayout_, 0, 1, &frame.textureDescriptor,
                            0, nullptr);
    struct alignas(16) CameraPush {
        std::array<float, 4> view;
        std::array<float, 2> worldSize;
        std::uint32_t debugMode;
        float time;
        std::array<float, 4> atmosphere;
        std::array<float, 4> celestial;
        std::array<float, 4> materialSurface;
        std::array<float, 4> materialDetail;
        std::array<float, 4> fluidSurface;
        std::array<float, 4> fluidMotion;
    };
    static_assert(sizeof(CameraPush) == cellPushConstantBytes);

    const CameraPush cameraPush{
        .view = {
            camera.x,
            camera.y,
            static_cast<float>(World::viewWidth),
            static_cast<float>(World::viewHeight),
        },
        .worldSize = {
            static_cast<float>(World::width),
            static_cast<float>(World::height),
        },
        .debugMode =
            rayTracingSettings_.debugDisplayMode > 0
                ? static_cast<std::uint32_t>(
                      rayTracingSettings_.debugDisplayMode + 1)
                : (world.liquidDebug() ? 1U : 0U),
        .time = static_cast<float>(SDL_GetTicks64()) * 0.001F,
        .atmosphere = {
            rayTracingSettings_.ambientIntensity +
                celestial.daylightFactor *
                    rayTracingSettings_.daylightAmbientIntensity,
            rayTracingSettings_.baseHaze,
            rayTracingSettings_.smokeHazeContribution,
            rayTracingSettings_.hazeAttenuation,
        },
        .celestial = {
            celestial.phase,
            celestial.daylight,
            rayTracingSettings_.skyIntensity,
            rayTracingSettings_.skyLightIntensity *
                celestial.daylightFactor,
        },
        .materialSurface = {
            rayTracingSettings_.materialSpecularStrength,
            rayTracingSettings_.materialNormalDetail,
            rayTracingSettings_.marblePolish,
            rayTracingSettings_.marbleVeinReflectivity,
        },
        .materialDetail = {
            rayTracingSettings_.marbleFleckDensity,
            rayTracingSettings_.marbleFleckReflectivity,
            packNormalizedPair(
                rayTracingSettings_.marbleSubsurfaceStrength, 3.0F,
                rayTracingSettings_.marbleScatterDistance, 12.0F),
            packNormalizedPair(
                rayTracingSettings_.liquidReflectionStrength, 3.0F,
                rayTracingSettings_.liquidSubsurfaceStrength, 3.0F),
        },
        .fluidSurface = {
            rayTracingSettings_.liquidMetaballs ? 1.0F : 0.0F,
            rayTracingSettings_.liquidMetaballDensity,
            rayTracingSettings_.liquidMetaballRadius,
            rayTracingSettings_.liquidMetaballEdgeSoftness,
        },
        .fluidMotion = {
            rayTracingSettings_.liquidFlowStretch,
            rayTracingSettings_.liquidPoolFlattening,
            rayTracingSettings_.liquidMetaballNormalStrength,
            packNormalizedPair(
                rayTracingSettings_.liquidCausticStrength, 3.0F,
                0.0F, 1.0F),
        },
    };
    vkCmdPushConstants(commandBuffer, cellPipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(cameraPush), &cameraPush);
    vkCmdDraw(commandBuffer, 6, 1, 0, 0);

    if (vertexCount > 0) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout_, 0, 1, &frame.textureDescriptor, 0,
            nullptr);
        vkCmdPushConstants(commandBuffer, pipelineLayout_,
                           VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(cameraPush), &cameraPush);
        const VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &frame.vertexBuffer, &offset);
        vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      particlePipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        particlePipelineLayout_, 0, 1, &frame.textureDescriptor, 0,
        nullptr);
    vkCmdPushConstants(commandBuffer, particlePipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(cameraPush), &cameraPush);
    vkCmdDraw(commandBuffer, 6, maximumGpuParticles, 0, 0);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
    vkCmdEndRenderPass(commandBuffer);
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    frame.lightingOriginX = lightingOriginX;
    frame.lightingOriginY = lightingOriginY;
    frame.lightingHistoryValid = true;
    frame.rayTimingQueriesValid = true;
    ++lightingFrameIndex_;
}

void VulkanRenderer::draw(World& world) {
    // Material state is persistent on the CPU-facing world grid. Consume the
    // previous compute result before uploading the next active region.
    synchronizeMaterialSimulation(world);
    FrameResources& frame = frames_[currentFrame_];
    check(vkWaitForFences(device_, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    if (frame.rayTimingQueriesValid) {
        std::array<std::uint64_t, rayTimingQueryCount> timestamps{};
        const VkResult timingResult = vkGetQueryPoolResults(
            device_, frame.rayTimingQueryPool, 0, rayTimingQueryCount,
            sizeof(timestamps), timestamps.data(),
            sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT);
        if (timingResult == VK_SUCCESS) {
            const auto milliseconds =
                [&](std::uint32_t start, std::uint32_t end) {
                    return static_cast<float>(
                        static_cast<double>(
                            timestamps[end] - timestamps[start]) *
                        static_cast<double>(
                            timestampPeriodNanoseconds_) /
                        1'000'000.0);
                };
            const GpuRayTimings sample{
                .derivedFieldsMs = milliseconds(
                    derivedTimingStart, derivedTimingEnd),
                .directLightingMs = milliseconds(
                    directTimingStart, directTimingEnd),
                .globalIlluminationMs = milliseconds(
                    giTimingStart, giTimingEnd),
                .denoisingMs = milliseconds(
                    denoiseTimingStart, denoiseTimingEnd),
                .totalMs = milliseconds(
                    rayTimingStart, rayTimingEnd),
                .valid = true,
            };
            constexpr float timingBlend = 0.10F;
            if (!gpuRayTimings_.valid) {
                gpuRayTimings_ = sample;
            } else {
                gpuRayTimings_.derivedFieldsMs +=
                    (sample.derivedFieldsMs -
                     gpuRayTimings_.derivedFieldsMs) *
                    timingBlend;
                gpuRayTimings_.directLightingMs +=
                    (sample.directLightingMs -
                     gpuRayTimings_.directLightingMs) *
                    timingBlend;
                gpuRayTimings_.globalIlluminationMs +=
                    (sample.globalIlluminationMs -
                     gpuRayTimings_.globalIlluminationMs) *
                    timingBlend;
                gpuRayTimings_.denoisingMs +=
                    (sample.denoisingMs -
                     gpuRayTimings_.denoisingMs) *
                    timingBlend;
                gpuRayTimings_.totalMs +=
                    (sample.totalMs - gpuRayTimings_.totalMs) *
                    timingBlend;
            }
        }
    }

    std::uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE,
        &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        check(acquireResult, "vkAcquireNextImageKHR");
    }

    const std::uint64_t particleTicks = SDL_GetTicks64();
    if (lastDayCycleTicks_ != 0 &&
        rayTracingSettings_.dayNightCycle) {
        const float cycleDeltaSeconds = std::clamp(
            static_cast<float>(
                particleTicks - lastDayCycleTicks_) *
                0.001F,
            0.0F, 0.10F);
        const float dayLength = std::max(
            rayTracingSettings_.dayLengthSeconds, 1.0F);
        rayTracingSettings_.timeOfDayHours =
            std::fmod(
                rayTracingSettings_.timeOfDayHours +
                    cycleDeltaSeconds * 24.0F / dayLength,
                24.0F);
    }
    lastDayCycleTicks_ = particleTicks;
    const Vec2 camera = world.renderCameraTopLeft();
    const CelestialState currentCelestial =
        celestialState(rayTracingSettings_);
    const std::int32_t currentLightingOriginX =
        static_cast<std::int32_t>(std::floor(camera.x));
    const std::int32_t currentLightingOriginY =
        static_cast<std::int32_t>(std::floor(camera.y));
    const float transmittanceDirectionSimilarity =
        sunTransmittanceCacheValid_
            ? sunTransmittanceDirection_.x *
                      currentCelestial.sunDirection.x +
                  sunTransmittanceDirection_.y *
                      currentCelestial.sunDirection.y
            : -1.0F;
    const std::int32_t requestedSunRayCount =
        std::clamp(rayTracingSettings_.sunRays, 1, 16);
    const bool transmittanceSettingsChanged =
        !sunTransmittanceCacheValid_ ||
        sunTransmittanceSmoke_ !=
            rayTracingSettings_.smokeTransmission ||
        sunTransmittanceSteam_ !=
            rayTracingSettings_.steamTransmission ||
        sunTransmittanceSoftness_ !=
            rayTracingSettings_.sunShadowSoftness ||
        sunTransmittanceRayCount_ != requestedSunRayCount;
    const bool transmittanceReprojectionExpired =
        !sunTransmittanceCacheValid_ ||
        std::abs(currentLightingOriginX -
                 sunTransmittanceOriginX_) >=
            static_cast<std::int32_t>(lightingWidth) ||
        std::abs(currentLightingOriginY -
                 sunTransmittanceOriginY_) >=
            static_cast<std::int32_t>(lightingHeight);
    const bool transmittanceRefreshDue =
        !sunTransmittanceCacheValid_ ||
        particleTicks - lastSunTransmittanceTicks_ >= 33;
    const bool transmittanceDirectionJumped =
        !sunTransmittanceCacheValid_ ||
        transmittanceDirectionSimilarity < 0.999F;
    updateSunTransmittanceThisFrame_ =
        currentCelestial.daylight > 0.0001F &&
        (transmittanceSettingsChanged ||
         transmittanceDirectionJumped ||
         transmittanceReprojectionExpired ||
         transmittanceRefreshDue);
    if (currentCelestial.daylight <= 0.0001F) {
        sunTransmittanceCacheValid_ = false;
    } else if (updateSunTransmittanceThisFrame_) {
        sunTransmittanceOriginX_ = currentLightingOriginX;
        sunTransmittanceOriginY_ = currentLightingOriginY;
        sunTransmittanceDirection_ =
            currentCelestial.sunDirection;
        sunTransmittanceSmoke_ =
            rayTracingSettings_.smokeTransmission;
        sunTransmittanceSteam_ =
            rayTracingSettings_.steamTransmission;
        sunTransmittanceSoftness_ =
            rayTracingSettings_.sunShadowSoftness;
        sunTransmittanceRayCount_ = requestedSunRayCount;
        lastSunTransmittanceTicks_ = particleTicks;
        sunTransmittanceCacheValid_ = true;
    }
    const float cachedDirectionLength =
        length(cachedSunDirection_);
    const float directionSimilarity =
        cachedDirectionLength > 0.0001F
            ? cachedSunDirection_.x *
                      currentCelestial.sunDirection.x +
                  cachedSunDirection_.y *
                      currentCelestial.sunDirection.y
            : -1.0F;
    const bool sunDirectionChanged =
        !sunVisibilityCacheValid_ ||
        directionSimilarity < 0.999998F;
    const bool solidFieldChanged =
        cachedSunSolidRevision_ != world.solidRevision();
    const bool sunReceiverMoved =
        !sunVisibilityCacheValid_ ||
        std::abs(camera.x - cachedSunCamera_.x) >
            static_cast<float>(World::viewWidth) * 0.25F ||
        std::abs(camera.y - cachedSunCamera_.y) >
            static_cast<float>(World::viewHeight) * 0.25F;
    const bool sunCacheThrottleExpired =
        !sunVisibilityCacheValid_ ||
        particleTicks - lastSunOcclusionTicks_ >= 33;
    if ((!sunVisibilityCacheValid_ ||
         sunDirectionChanged || solidFieldChanged ||
         sunReceiverMoved) &&
        sunCacheThrottleExpired) {
        const Vec2 receiverMargin{
            static_cast<float>(World::viewWidth) * 0.5F,
            static_cast<float>(World::viewHeight) * 0.5F,
        };
        world.buildDirectionalSunHorizon(
            currentCelestial.sunDirection,
            sunHorizonSamplesPerCell,
            sunHorizonDepths_,
            sunHorizonBlockers_,
            sunHorizonMinimum_,
            camera - receiverMargin,
            camera + Vec2{
                static_cast<float>(World::viewWidth),
                static_cast<float>(World::viewHeight),
            } + receiverMargin);
        cachedSunDirection_ = currentCelestial.sunDirection;
        cachedSunCamera_ = camera;
        cachedSunSolidRevision_ = world.solidRevision();
        lastSunOcclusionTicks_ = particleTicks;
        sunVisibilityCacheValid_ = true;
    }
    if (lastParticleTicks_ != 0) {
        particleDeltaTime_ = std::clamp(
            static_cast<float>(particleTicks - lastParticleTicks_) * 0.001F,
            0.0F, 0.05F);
    }
    lastParticleTicks_ = particleTicks;
    uploadParticleSpawns(frame, world);
    prepareMaterialSimulation(world);

    updateSceneLights(world, camera);
    GpuSceneLights sceneLights{};
    sceneLights.playerLight = playerLight_;
    for (const auto& light : fireLights_) {
        if (light[2] <= 0.0F || light[3] <= 0.001F) {
            continue;
        }
        const std::uint32_t lightIndex = sceneLights.counts[0]++;
        sceneLights.fireLights[lightIndex] = light;
    }
    void* sceneLightMapped = nullptr;
    check(vkMapMemory(device_, frame.sceneLightMemory, 0,
                      sizeof(sceneLights), 0, &sceneLightMapped),
          "vkMapMemory scene lights");
    std::memcpy(sceneLightMapped, &sceneLights, sizeof(sceneLights));
    vkUnmapMemory(device_, frame.sceneLightMemory);

    void* lightTilesMapped = nullptr;
    const VkDeviceSize lightTileBytes =
        static_cast<VkDeviceSize>(lightTileCount) *
        sizeof(GpuLightTile);
    check(vkMapMemory(device_, frame.lightTileMemory, 0,
                      lightTileBytes, 0, &lightTilesMapped),
          "vkMapMemory light tiles");
    auto* lightTiles =
        static_cast<GpuLightTile*>(lightTilesMapped);
    for (std::uint32_t tile = 0; tile < lightTileCount; ++tile) {
        lightTiles[tile].count = 0;
    }
    const float lightingOriginX = std::floor(camera.x);
    const float lightingOriginY = std::floor(camera.y);
    for (std::uint32_t lightIndex = 0;
         lightIndex < sceneLights.counts[0]; ++lightIndex) {
        const auto& light = sceneLights.fireLights[lightIndex];
        const float radius = light[2];
        const int minimumTileX = std::clamp(
            static_cast<int>(std::floor(
                (light[0] - radius - lightingOriginX) /
                static_cast<float>(lightTileSize))),
            0, static_cast<int>(lightTileColumns) - 1);
        const int maximumTileX = std::clamp(
            static_cast<int>(std::floor(
                (light[0] + radius - lightingOriginX) /
                static_cast<float>(lightTileSize))),
            0, static_cast<int>(lightTileColumns) - 1);
        const int minimumTileY = std::clamp(
            static_cast<int>(std::floor(
                (light[1] - radius - lightingOriginY) /
                static_cast<float>(lightTileSize))),
            0, static_cast<int>(lightTileRows) - 1);
        const int maximumTileY = std::clamp(
            static_cast<int>(std::floor(
                (light[1] + radius - lightingOriginY) /
                static_cast<float>(lightTileSize))),
            0, static_cast<int>(lightTileRows) - 1);
        for (int tileY = minimumTileY;
             tileY <= maximumTileY; ++tileY) {
            for (int tileX = minimumTileX;
                 tileX <= maximumTileX; ++tileX) {
                GpuLightTile& tile =
                    lightTiles[
                        static_cast<std::size_t>(tileY) *
                            lightTileColumns +
                        static_cast<std::size_t>(tileX)];
                tile.indices[tile.count++] = lightIndex;
            }
        }
    }
    vkUnmapMemory(device_, frame.lightTileMemory);

    const std::vector<Vertex> vertices = buildVertices(world);
    const VkDeviceSize byteCount =
        static_cast<VkDeviceSize>(vertices.size() * sizeof(Vertex));
    if (byteCount > vertexBufferBytes) {
        throw std::runtime_error("The generated world exceeds the vertex buffer");
    }
    void* mapped = nullptr;
    check(vkMapMemory(device_, frame.vertexMemory, 0, byteCount, 0, &mapped),
          "vkMapMemory");
    std::memcpy(mapped, vertices.data(), static_cast<std::size_t>(byteCount));
    vkUnmapMemory(device_, frame.vertexMemory);

    const VkDeviceSize textureBytes =
        static_cast<VkDeviceSize>(textureWidth) *
        static_cast<VkDeviceSize>(textureHeight) * 4U;
    void* textureMapped = nullptr;
    check(vkMapMemory(device_, frame.textureStagingMemory, 0, textureBytes, 0,
                      &textureMapped),
          "vkMapMemory");
    auto* texturePixels = static_cast<std::uint8_t*>(textureMapped);
    GpuOccupancyHierarchy occupancy{};
    const int originX = static_cast<int>(std::floor(camera.x));
    const int originY = static_cast<int>(std::floor(camera.y));
    std::array<int, textureWidth + 2U> visibleSkyOccluders{};
    for (std::size_t index = 0;
         index < visibleSkyOccluders.size(); ++index) {
        visibleSkyOccluders[index] =
            world.skyOccluderY(
                originX + static_cast<int>(index) - 1);
    }
    const auto sunBlockedAt =
        [&](float sampleX, float sampleY,
            std::int32_t receiverIndex) {
            const float perpendicular =
                -cachedSunDirection_.y * sampleX +
                cachedSunDirection_.x * sampleY;
            const int sample = static_cast<int>(std::floor(
                (perpendicular - sunHorizonMinimum_) *
                sunHorizonSamplesPerCell));
            if (sample < 0 ||
                sample >=
                    static_cast<int>(sunHorizonBlockers_.size())) {
                return false;
            }
            const std::size_t sampleIndex =
                static_cast<std::size_t>(sample);
            const std::int32_t blocker =
                sunHorizonBlockers_[sampleIndex];
            if (blocker < 0 || blocker == receiverIndex) {
                return false;
            }
            const float receiverDepth =
                cachedSunDirection_.x * sampleX +
                cachedSunDirection_.y * sampleY;
            return sunHorizonDepths_[sampleIndex] >
                   receiverDepth + 0.001F;
        };

    for (std::uint32_t textureY = 0; textureY < textureHeight; ++textureY) {
        const int worldY = std::clamp(
            originY + static_cast<int>(textureY), 0, World::height - 1);
        const std::size_t rowStart =
            static_cast<std::size_t>(worldY) * World::width;
        for (std::uint32_t textureX = 0; textureX < textureWidth; ++textureX) {
            const int worldX = std::clamp(
                originX + static_cast<int>(textureX), 0, World::width - 1);
            const std::size_t worldIndex =
                rowStart + static_cast<std::size_t>(worldX);
            const std::size_t textureIndex =
                static_cast<std::size_t>(textureY * textureWidth + textureX) *
                4U;
            const Material material = world.materials()[worldIndex];
            if (isSunOccluder(material) ||
                material == Material::water ||
                material == Material::oil) {
                const std::uint32_t blockX =
                    textureX / occupancyBlockSize;
                const std::uint32_t blockY =
                    textureY / occupancyBlockSize;
                occupancy.occupied[
                    static_cast<std::size_t>(blockY) *
                        occupancyColumns +
                    blockX] = 1U;
            }
            texturePixels[textureIndex] = static_cast<std::uint8_t>(material);
            texturePixels[textureIndex + 1] = 0;
            texturePixels[textureIndex + 2] = 0;
            const bool receiverIsSolid =
                isSunOccluder(material);
            const float centerX =
                static_cast<float>(worldX) + 0.5F;
            const float centerY =
                static_cast<float>(worldY) + 0.5F;
            bool sunlit = false;
            if (!receiverIsSolid) {
                sunlit = !sunBlockedAt(
                    centerX, centerY,
                    static_cast<std::int32_t>(worldIndex));
            } else {
                // Shade opaque cells from their exposed sun-facing faces.
                // Sampling a solid at its center makes a long roof or wall
                // block itself at oblique angles, producing crawling stripes.
                // Moving the lookup just beyond an exposed face represents
                // the surface that is actually visible in this 2D cutaway.
                constexpr float directionEpsilon = 0.00001F;
                constexpr float quantizationClearance =
                    0.5F / sunHorizonSamplesPerCell;
                constexpr float faceOffset =
                    0.5F + quantizationClearance + 0.001F;
                const int sunwardX =
                    cachedSunDirection_.x > directionEpsilon
                        ? 1
                        : (cachedSunDirection_.x <
                                   -directionEpsilon
                               ? -1
                               : 0);
                const int sunwardY =
                    cachedSunDirection_.y > directionEpsilon
                        ? 1
                        : (cachedSunDirection_.y <
                                   -directionEpsilon
                               ? -1
                               : 0);
                const int horizontalNeighborX =
                    worldX + sunwardX;
                const bool horizontalNeighborSolid =
                    sunwardX == 0 ||
                    horizontalNeighborX < 0 ||
                    horizontalNeighborX >= World::width ||
                    isSunOccluder(
                        world.materials()[
                            rowStart +
                            static_cast<std::size_t>(
                                horizontalNeighborX)]);
                if (sunwardX != 0 &&
                    !horizontalNeighborSolid) {
                    sunlit = !sunBlockedAt(
                        centerX +
                            static_cast<float>(sunwardX) *
                                faceOffset,
                        centerY,
                        static_cast<std::int32_t>(worldIndex));
                }
                const int verticalNeighborY =
                    worldY + sunwardY;
                const bool verticalNeighborSolid =
                    sunwardY == 0 ||
                    verticalNeighborY < 0 ||
                    verticalNeighborY >= World::height ||
                    isSunOccluder(
                        world.materials()[
                            static_cast<std::size_t>(
                                verticalNeighborY) *
                                World::width +
                            static_cast<std::size_t>(worldX)]);
                if (!sunlit && sunwardY != 0 &&
                    !verticalNeighborSolid) {
                    sunlit = !sunBlockedAt(
                        centerX,
                        centerY +
                            static_cast<float>(sunwardY) *
                                faceOffset,
                        static_cast<std::int32_t>(worldIndex));
                }
            }
            const std::uint8_t worldSun =
                sunlit ? 255 : 0;
            const std::uint8_t packedSun7 =
                static_cast<std::uint8_t>(
                    (static_cast<unsigned>(worldSun) * 127U + 127U) /
                    255U);
            texturePixels[textureIndex + 3] = packedSun7;

            const std::size_t skyColumn =
                static_cast<std::size_t>(textureX) + 1U;
            const bool verticallyOpenToSky =
                worldY < visibleSkyOccluders[skyColumn] ||
                (receiverIsSolid &&
                 worldY == visibleSkyOccluders[skyColumn]);
            const bool leftOpenToSky =
                worldX > 0 &&
                worldY <
                    visibleSkyOccluders[skyColumn - 1U] &&
                !isSunOccluder(
                    world.materials()[worldIndex - 1U]);
            const bool rightOpenToSky =
                worldX + 1 < World::width &&
                worldY <
                    visibleSkyOccluders[skyColumn + 1U] &&
                !isSunOccluder(
                    world.materials()[worldIndex + 1U]);
            const bool skyLightExposed =
                verticallyOpenToSky ||
                leftOpenToSky || rightOpenToSky;
            const bool liquid =
                material == Material::water ||
                material == Material::oil;
            if (skyLightExposed && !liquid) {
                texturePixels[textureIndex + 3] |= 0x80U;
            }

            if (material == Material::air) {
                if (verticallyOpenToSky &&
                    !world.hasInteriorBackdrop(worldX, worldY)) {
                    // Air uses green as a separate visual-sky mask. The alpha
                    // high bit is diffuse sky-light exposure and deliberately
                    // remains available inside a breached tomb.
                    texturePixels[textureIndex + 1] = 255U;
                }
                continue;
            }
            if (material == Material::wood) {
                texturePixels[textureIndex + 1] =
                    static_cast<std::uint8_t>(
                        std::clamp(world.burnProgress()[worldIndex],
                                   0.0F, 1.0F) *
                        255.0F);
                texturePixels[textureIndex + 2] =
                    static_cast<std::uint8_t>(
                        std::clamp(world.heat()[worldIndex], 0.0F, 1.0F) *
                        255.0F);
                continue;
            }
            if (material != Material::water &&
                material != Material::oil) {
                continue;
            }

            texturePixels[textureIndex + 1] =
                static_cast<std::uint8_t>(std::min<std::uint16_t>(
                    world.liquidAmounts()[worldIndex], 255));
            const int flowX =
                static_cast<int>(world.liquidFlowX()[worldIndex]);
            const int flowY =
                static_cast<int>(world.liquidFlowY()[worldIndex]);
            const int packedFlowX =
                std::clamp((flowX + 127) * 15 / 254, 0, 15);
            const int packedFlowY =
                std::clamp((flowY + 127) * 15 / 254, 0, 15);
            texturePixels[textureIndex + 2] =
                static_cast<std::uint8_t>((packedFlowX << 4) |
                                          packedFlowY);
            const std::uint8_t packedFoam =
                static_cast<std::uint8_t>(
                    (static_cast<unsigned>(
                         world.liquidFoam()[worldIndex]) +
                     18U) /
                    36U);
            const std::uint8_t packedSun4 =
                static_cast<std::uint8_t>(
                    (static_cast<unsigned>(worldSun) + 8U) /
                    17U);
            texturePixels[textureIndex + 3] =
                static_cast<std::uint8_t>(
                    (skyLightExposed ? 0x80U : 0U) |
                    ((packedFoam & 0x07U) << 4U) |
                    packedSun4);
        }
    }
    vkUnmapMemory(device_, frame.textureStagingMemory);
    for (std::uint32_t largeY = 0;
         largeY < occupancyLargeRows; ++largeY) {
        for (std::uint32_t largeX = 0;
             largeX < occupancyLargeColumns; ++largeX) {
            bool occupied = false;
            for (std::uint32_t offsetY = 0;
                 offsetY < occupancyLargeBlockSize /
                                   occupancyBlockSize &&
                 !occupied;
                 ++offsetY) {
                for (std::uint32_t offsetX = 0;
                     offsetX < occupancyLargeBlockSize /
                                       occupancyBlockSize;
                     ++offsetX) {
                    const std::uint32_t blockX =
                        largeX *
                            (occupancyLargeBlockSize /
                             occupancyBlockSize) +
                        offsetX;
                    const std::uint32_t blockY =
                        largeY *
                            (occupancyLargeBlockSize /
                             occupancyBlockSize) +
                        offsetY;
                    if (blockX < occupancyColumns &&
                        blockY < occupancyRows &&
                        occupancy.occupied[
                            static_cast<std::size_t>(blockY) *
                                occupancyColumns +
                            blockX] != 0U) {
                        occupied = true;
                        break;
                    }
                }
            }
            occupancy.occupied[
                occupancyBlockCount +
                static_cast<std::size_t>(largeY) *
                    occupancyLargeColumns +
                largeX] = occupied ? 1U : 0U;
        }
    }
    void* occupancyMapped = nullptr;
    check(vkMapMemory(device_, frame.occupancyMemory, 0,
                      sizeof(occupancy), 0, &occupancyMapped),
          "vkMapMemory occupancy hierarchy");
    std::memcpy(occupancyMapped, &occupancy,
                sizeof(occupancy));
    vkUnmapMemory(device_, frame.occupancyMemory);

    check(vkResetFences(device_, 1, &frame.inFlight), "vkResetFences");
    check(vkResetCommandBuffer(frame.commandBuffer, 0), "vkResetCommandBuffer");
    recordCommands(frame.commandBuffer, imageIndex,
                   static_cast<std::uint32_t>(vertices.size()), world);

    constexpr VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    const VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame.imageAvailable,
        .pWaitDstStageMask = &waitStage,
        .commandBufferCount = 1,
        .pCommandBuffers = &frame.commandBuffer,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &frame.renderFinished,
    };
    check(vkQueueSubmit(graphicsQueue_, 1, &submitInfo, frame.inFlight),
          "vkQueueSubmit");
    if (materialSimulationSteps_ > 0) {
        materialSimulationPending_ = true;
        materialSimulationFrame_ = currentFrame_;
        materialSimulationSteps_ = 0;
    }

    const VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame.renderFinished,
        .swapchainCount = 1,
        .pSwapchains = &swapchain_,
        .pImageIndices = &imageIndex,
        .pResults = nullptr,
    };
    const VkResult presentResult = vkQueuePresentKHR(presentQueue_, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR ||
        presentResult == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else {
        check(presentResult, "vkQueuePresentKHR");
    }
    currentFrame_ = (currentFrame_ + 1) % framesInFlight;
}

void VulkanRenderer::destroySwapchain() {
    for (VkFramebuffer framebuffer : framebuffers_) {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    framebuffers_.clear();
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        pipelineLayout_ = VK_NULL_HANDLE;
    }
    if (cellPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, cellPipeline_, nullptr);
        cellPipeline_ = VK_NULL_HANDLE;
    }
    if (cellPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, cellPipelineLayout_, nullptr);
        cellPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (computePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, computePipeline_, nullptr);
        computePipeline_ = VK_NULL_HANDLE;
    }
    if (computePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, computePipelineLayout_, nullptr);
        computePipelineLayout_ = VK_NULL_HANDLE;
    }
    if (derivedPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, derivedPipeline_, nullptr);
        derivedPipeline_ = VK_NULL_HANDLE;
    }
    if (derivedPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, derivedPipelineLayout_, nullptr);
        derivedPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (particleComputePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, particleComputePipeline_, nullptr);
        particleComputePipeline_ = VK_NULL_HANDLE;
    }
    if (particleComputePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, particleComputePipelineLayout_,
                                nullptr);
        particleComputePipelineLayout_ = VK_NULL_HANDLE;
    }
    if (particlePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, particlePipeline_, nullptr);
        particlePipeline_ = VK_NULL_HANDLE;
    }
    if (particlePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, particlePipelineLayout_, nullptr);
        particlePipelineLayout_ = VK_NULL_HANDLE;
    }
    if (materialComputePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, materialComputePipeline_, nullptr);
        materialComputePipeline_ = VK_NULL_HANDLE;
    }
    if (materialComputePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, materialComputePipelineLayout_,
                                nullptr);
        materialComputePipelineLayout_ = VK_NULL_HANDLE;
    }
    if (renderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        renderPass_ = VK_NULL_HANDLE;
    }
    for (VkImageView view : swapchainImageViews_) {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchainImageViews_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::recreateSwapchain() {
    int width = 0;
    int height = 0;
    SDL_Vulkan_GetDrawableSize(window_, &width, &height);
    if (width == 0 || height == 0) {
        return;
    }
    vkDeviceWaitIdle(device_);
    destroySwapchain();
    createSwapchain();
    createRenderPass();
    createPipeline();
    createFramebuffers();
    if (imguiInitialized_) {
        ImGui_ImplVulkan_SetMinImageCount(2);
        ImGui_ImplVulkan_PipelineInfo pipelineInfo{};
        pipelineInfo.RenderPass = renderPass_;
        pipelineInfo.Subpass = 0;
        pipelineInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        ImGui_ImplVulkan_CreateMainPipeline(&pipelineInfo);
    }
}

void VulkanRenderer::waitIdle() const {
    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

void VulkanRenderer::refreshDisplay() {
    recreateSwapchain();
}

} // namespace gunpowder
