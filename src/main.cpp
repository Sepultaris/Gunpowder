#include "game/world.hpp"
#include "render/vulkan_renderer.hpp"

#include <SDL.h>
#include <imgui.h>
#include <imgui_impl_sdl2.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class SdlContext {
public:
    SdlContext() {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
            throw std::runtime_error(SDL_GetError());
        }
    }
    ~SdlContext() { SDL_Quit(); }

    SdlContext(const SdlContext&) = delete;
    SdlContext& operator=(const SdlContext&) = delete;
};

struct WindowDeleter {
    void operator()(SDL_Window* window) const { SDL_DestroyWindow(window); }
};

struct DisplayResolution {
    int width;
    int height;
    const char* label;
};

constexpr std::array displayResolutions{
    DisplayResolution{1280, 720, "1280 x 720"},
    DisplayResolution{1600, 900, "1600 x 900"},
    DisplayResolution{1920, 1080, "1920 x 1080"},
    DisplayResolution{2560, 1440, "2560 x 1440"},
    DisplayResolution{3840, 2160, "3840 x 2160"},
};

enum class WindowMode : int {
    windowed = 0,
    borderlessWindow = 1,
    borderlessFullscreen = 2,
    exclusiveFullscreen = 3,
};

struct DisplaySettings {
    int resolutionIndex = 2;
    int windowMode = static_cast<int>(WindowMode::borderlessWindow);
};

#define GUNPOWDER_RENDER_PROFILE_FIELDS(X)                                  \
    X(raysPerLight)                                                         \
    X(softShadowRadius)                                                     \
    X(solidTransmission)                                                    \
    X(smokeTransmission)                                                    \
    X(steamTransmission)                                                    \
    X(indirectBounces)                                                      \
    X(giRays)                                                               \
    X(giIntensity)                                                          \
    X(bounceDistance)                                                       \
    X(temporalGiDenoising)                                                  \
    X(giHistoryWeight)                                                      \
    X(adaptiveGiSampling)                                                   \
    X(debugDisplayMode)                                                     \
    X(playerLightRadius)                                                    \
    X(playerLightIntensity)                                                 \
    X(fireBaseRadius)                                                       \
    X(fireClusterRadiusScale)                                               \
    X(fireMaximumRadiusBonus)                                               \
    X(fireBaseIntensity)                                                    \
    X(fireClusterIntensityScale)                                            \
    X(fireMaximumIntensity)                                                 \
    X(dayNightCycle)                                                        \
    X(timeOfDayHours)                                                       \
    X(dayLengthSeconds)                                                     \
    X(sunRays)                                                              \
    X(sunIntensity)                                                         \
    X(sunShadowSoftness)                                                    \
    X(skyIntensity)                                                         \
    X(skyLightIntensity)                                                    \
    X(ambientIntensity)                                                     \
    X(daylightAmbientIntensity)                                             \
    X(baseHaze)                                                             \
    X(smokeHazeContribution)                                                \
    X(hazeAttenuation)                                                      \
    X(materialSpecularStrength)                                             \
    X(materialNormalDetail)                                                 \
    X(marblePolish)                                                         \
    X(marbleVeinReflectivity)                                               \
    X(marbleFleckDensity)                                                   \
    X(marbleFleckReflectivity)                                              \
    X(marbleSubsurfaceStrength)                                             \
    X(marbleScatterDistance)                                                \
    X(liquidMetaballs)                                                      \
    X(liquidMetaballDensity)                                                \
    X(liquidMetaballRadius)                                                 \
    X(liquidMetaballEdgeSoftness)                                           \
    X(liquidFlowStretch)                                                    \
    X(liquidPoolFlattening)                                                 \
    X(liquidMetaballNormalStrength)                                         \
    X(liquidReflectionStrength)                                             \
    X(liquidSubsurfaceStrength)                                             \
    X(liquidCausticStrength)                                                \
    X(liquidDispersionStrength)

struct RenderProfileUi {
    std::filesystem::path directory;
    std::vector<std::string> names;
    int selected = -1;
    std::array<char, 64> name{};
    std::string status;
    std::string error;

    explicit RenderProfileUi(std::filesystem::path profileDirectory)
        : directory(std::move(profileDirectory)) {
        constexpr std::string_view initialName = "My Look";
        std::copy(initialName.begin(), initialName.end(), name.begin());
    }
};

std::filesystem::path renderProfileDirectory() {
    char* preferencePath = SDL_GetPrefPath("Sepul", "Gunpowder");
    if (preferencePath == nullptr) {
        return std::filesystem::current_path() / "render-profiles";
    }
    std::filesystem::path result(preferencePath);
    SDL_free(preferencePath);
    return result / "render-profiles";
}

std::string cleanProfileName(std::string_view rawName) {
    std::string result;
    result.reserve(std::min<std::size_t>(rawName.size(), 56));
    bool previousSpace = false;
    for (const char character : rawName) {
        if (result.size() >= 56) {
            break;
        }
        const unsigned char value =
            static_cast<unsigned char>(character);
        if (std::isalnum(value) != 0 ||
            character == '-' || character == '_') {
            result.push_back(character);
            previousSpace = false;
        } else if (std::isspace(value) != 0 && !result.empty() &&
                   !previousSpace) {
            result.push_back(' ');
            previousSpace = true;
        }
    }
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result.empty() ? "Untitled Look" : result;
}

constexpr std::string_view profileSuffix = ".render.ini";

std::filesystem::path profilePath(
    const std::filesystem::path& directory,
    std::string_view profileName) {
    return directory /
           (cleanProfileName(profileName) +
            std::string(profileSuffix));
}

void refreshRenderProfiles(RenderProfileUi& profiles) {
    const std::string previouslySelected =
        profiles.selected >= 0 &&
                profiles.selected <
                    static_cast<int>(profiles.names.size())
            ? profiles.names[static_cast<std::size_t>(
                  profiles.selected)]
            : std::string{};
    profiles.names.clear();
    profiles.selected = -1;
    std::error_code error;
    std::filesystem::create_directories(
        profiles.directory, error);
    if (error) {
        profiles.error =
            "Could not create profile directory: " +
            error.message();
        return;
    }
    for (const auto& entry :
         std::filesystem::directory_iterator(
             profiles.directory, error)) {
        if (error || !entry.is_regular_file()) {
            continue;
        }
        const std::string filename =
            entry.path().filename().string();
        if (filename.size() <= profileSuffix.size() ||
            filename.compare(
                filename.size() - profileSuffix.size(),
                profileSuffix.size(), profileSuffix) != 0) {
            continue;
        }
        profiles.names.push_back(filename.substr(
            0, filename.size() - profileSuffix.size()));
    }
    std::sort(profiles.names.begin(), profiles.names.end());
    if (!previouslySelected.empty()) {
        const auto found = std::find(
            profiles.names.begin(), profiles.names.end(),
            previouslySelected);
        if (found != profiles.names.end()) {
            profiles.selected = static_cast<int>(
                std::distance(profiles.names.begin(), found));
        }
    }
    if (profiles.selected < 0 && !profiles.names.empty()) {
        profiles.selected = 0;
    }
    profiles.error.clear();
}

template <typename Value>
void writeProfileValue(std::ostream& stream, const char* key,
                       const Value& value) {
    stream << key << '=';
    if constexpr (std::is_same_v<Value, bool>) {
        stream << (value ? "true" : "false");
    } else if constexpr (std::is_floating_point_v<Value>) {
        stream << std::setprecision(
                      std::numeric_limits<Value>::max_digits10)
               << value;
    } else {
        stream << value;
    }
    stream << '\n';
}

bool saveRenderProfile(
    const std::filesystem::path& directory,
    std::string_view profileName,
    const gunpowder::RayTracingSettings& settings,
    std::string& errorMessage) {
    std::error_code directoryError;
    std::filesystem::create_directories(
        directory, directoryError);
    if (directoryError) {
        errorMessage =
            "Could not create profile directory: " +
            directoryError.message();
        return false;
    }
    const std::filesystem::path path =
        profilePath(directory, profileName);
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        errorMessage =
            "Could not open " + path.string() + " for writing.";
        return false;
    }
    file << "# Gunpowder rendering profile\n";
    file << "version=1\n";
#define WRITE_RENDER_PROFILE_FIELD(field) \
    writeProfileValue(file, #field, settings.field);
    GUNPOWDER_RENDER_PROFILE_FIELDS(
        WRITE_RENDER_PROFILE_FIELD)
#undef WRITE_RENDER_PROFILE_FIELD
    if (!file) {
        errorMessage =
            "Writing " + path.string() + " failed.";
        return false;
    }
    errorMessage.clear();
    return true;
}

template <typename Value>
bool parseProfileValue(std::string_view text, Value& value) {
    if constexpr (std::is_same_v<Value, bool>) {
        if (text == "true" || text == "1") {
            value = true;
            return true;
        }
        if (text == "false" || text == "0") {
            value = false;
            return true;
        }
        return false;
    } else {
        std::istringstream stream{std::string(text)};
        Value parsed{};
        stream >> parsed;
        if (stream.fail()) {
            return false;
        }
        std::string trailingText;
        if (stream >> trailingText) {
            return false;
        }
        if constexpr (std::is_floating_point_v<Value>) {
            if (!std::isfinite(parsed)) {
                return false;
            }
        }
        value = parsed;
        return true;
    }
}

bool loadRenderProfile(
    const std::filesystem::path& directory,
    std::string_view profileName,
    gunpowder::RayTracingSettings& settings,
    std::string& errorMessage) {
    const std::filesystem::path path =
        profilePath(directory, profileName);
    std::ifstream file(path);
    if (!file) {
        errorMessage =
            "Could not open " + path.string() + ".";
        return false;
    }
    std::unordered_map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t first =
            line.find_first_not_of(" \t\r");
        if (first == std::string::npos ||
            line[first] == '#') {
            continue;
        }
        const std::size_t separator = line.find('=', first);
        if (separator == std::string::npos) {
            continue;
        }
        std::string key =
            line.substr(first, separator - first);
        const std::size_t keyEnd =
            key.find_last_not_of(" \t");
        key.erase(keyEnd + 1);
        std::string value = line.substr(separator + 1);
        const std::size_t valueFirst =
            value.find_first_not_of(" \t");
        value.erase(
            0, valueFirst == std::string::npos
                   ? value.size()
                   : valueFirst);
        const std::size_t valueEnd =
            value.find_last_not_of(" \t\r");
        if (valueEnd != std::string::npos) {
            value.erase(valueEnd + 1);
        }
        values[std::move(key)] = std::move(value);
    }

    gunpowder::RayTracingSettings loaded = settings;
#define LOAD_RENDER_PROFILE_FIELD(field)                                  \
    if (const auto found = values.find(#field); found != values.end()) {  \
        if (!parseProfileValue(found->second, loaded.field)) {            \
            errorMessage =                                                \
                "Invalid value for '" #field "' in " + path.string();     \
            return false;                                                 \
        }                                                                 \
    }
    GUNPOWDER_RENDER_PROFILE_FIELDS(
        LOAD_RENDER_PROFILE_FIELD)
#undef LOAD_RENDER_PROFILE_FIELD
    settings = loaded;
    errorMessage.clear();
    return true;
}

#undef GUNPOWDER_RENDER_PROFILE_FIELDS

std::string applyDisplaySettings(SDL_Window* window,
                                 const DisplaySettings& settings) {
    const DisplayResolution& resolution =
        displayResolutions[static_cast<std::size_t>(
            std::clamp(settings.resolutionIndex, 0,
                       static_cast<int>(displayResolutions.size()) - 1))];
    const WindowMode mode = static_cast<WindowMode>(
        std::clamp(settings.windowMode, 0, 3));

    if (SDL_SetWindowFullscreen(window, 0) != 0) {
        return SDL_GetError();
    }
    if (SDL_SetWindowDisplayMode(window, nullptr) != 0) {
        return SDL_GetError();
    }

    if (mode == WindowMode::borderlessFullscreen) {
        SDL_SetWindowBordered(window, SDL_FALSE);
        if (SDL_SetWindowFullscreen(window,
                                    SDL_WINDOW_FULLSCREEN_DESKTOP) != 0) {
            return SDL_GetError();
        }
        return {};
    }

    if (mode == WindowMode::exclusiveFullscreen) {
        const int displayIndex =
            std::max(SDL_GetWindowDisplayIndex(window), 0);
        SDL_DisplayMode desired{
            .format = SDL_PIXELFORMAT_UNKNOWN,
            .w = resolution.width,
            .h = resolution.height,
            .refresh_rate = 0,
            .driverdata = nullptr,
        };
        SDL_DisplayMode closest{};
        if (SDL_GetClosestDisplayMode(displayIndex, &desired, &closest) ==
            nullptr) {
            return SDL_GetError();
        }
        if (SDL_SetWindowDisplayMode(window, &closest) != 0 ||
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN) != 0) {
            return SDL_GetError();
        }
        return {};
    }

    SDL_SetWindowBordered(
        window, mode == WindowMode::windowed ? SDL_TRUE : SDL_FALSE);
    SDL_SetWindowResizable(window, SDL_FALSE);
    SDL_SetWindowSize(window, resolution.width, resolution.height);
    SDL_SetWindowPosition(window, SDL_WINDOWPOS_CENTERED,
                         SDL_WINDOWPOS_CENTERED);
    SDL_RaiseWindow(window);
    return {};
}

const char* materialName(gunpowder::Material material) {
    switch (material) {
    case gunpowder::Material::sand:
        return "Sand";
    case gunpowder::Material::water:
        return "Water";
    case gunpowder::Material::oil:
        return "Oil";
    case gunpowder::Material::fire:
        return "Fire";
    case gunpowder::Material::smoke:
        return "Smoke";
    case gunpowder::Material::wood:
        return "Wood";
    case gunpowder::Material::metal:
        return "Metal";
    default:
        return "Material";
    }
}

void sliderIntWithReset(const char* label, int* value, int minimum,
                        int maximum, int defaultValue,
                        ImGuiSliderFlags flags = 0) {
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(280.0F);
    ImGui::SliderInt("##value", value, minimum, maximum, "%d", flags);
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset")) {
        *value = defaultValue;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(label);
    ImGui::PopID();
}

void sliderFloatWithReset(const char* label, float* value, float minimum,
                          float maximum, const char* format,
                          float defaultValue,
                          ImGuiSliderFlags flags = 0) {
    ImGui::PushID(label);
    ImGui::SetNextItemWidth(280.0F);
    ImGui::SliderFloat("##value", value, minimum, maximum, format, flags);
    ImGui::SameLine();
    if (ImGui::SmallButton("Reset")) {
        *value = defaultValue;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(label);
    ImGui::PopID();
}

bool drawDeveloperUi(bool* open,
                     gunpowder::RayTracingSettings& settings,
                     const gunpowder::GpuRayTimings& gpuTimings,
                     const gunpowder::MaterialSimulationTimings&
                         materialTimings,
                     DisplaySettings& displaySettings,
                     RenderProfileUi& profiles,
                     const std::string& displayError,
                     float framesPerSecond) {
    const gunpowder::RayTracingSettings defaults{};
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(620.0F, 520.0F), ImVec2(2000.0F, 1200.0F));
    ImGui::SetNextWindowSize(ImVec2(620.0F, 720.0F),
                             ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Developer Settings", open)) {
        ImGui::End();
        return false;
    }

    ImGui::TextUnformatted(
        "Rendering settings apply immediately; display changes use Apply.");
    ImGui::Text("Performance: %.0f FPS", framesPerSecond);
    if (gpuTimings.valid) {
        ImGui::Text(
            "GPU lighting: %.3f ms", gpuTimings.totalMs);
        ImGui::TextDisabled(
            "Derived %.3f | Direct %.3f | GI %.3f | Denoise %.3f ms",
            gpuTimings.derivedFieldsMs,
            gpuTimings.directLightingMs,
            gpuTimings.globalIlluminationMs,
            gpuTimings.denoisingMs);
    } else {
        ImGui::TextDisabled("GPU lighting timings: collecting...");
    }
    if (materialTimings.valid) {
        ImGui::Text(
            "CPU materials: %.3f ms", materialTimings.totalMs);
        ImGui::TextDisabled(
            "Granular %.3f | Prepare %.3f | Flow %.3f | "
            "Gas %.3f | Heat %.3f ms",
            materialTimings.granularMs,
            materialTimings.liquidPreparationMs,
            materialTimings.liquidTransportMs,
            materialTimings.gasAndReactionMs,
            materialTimings.heatMs);
        ImGui::TextDisabled(
            "Flow: Select %.3f | Gravity %.3f | Lateral %.3f | "
            "Frontier %.3f | Equalize %.3f ms",
            materialTimings.liquidSelectionMs,
            materialTimings.liquidGravityMs,
            materialTimings.liquidLateralMs,
            materialTimings.liquidFrontierMs,
            materialTimings.liquidEqualizationMs);
        ImGui::TextDisabled(
            "Active chunks %u | Liquid candidates %u | "
            "Equalized %u components / %u cells",
            materialTimings.activeChunks,
            materialTimings.liquidCandidateVisits,
            materialTimings.equalizedComponents,
            materialTimings.equalizedCells);
        ImGui::TextDisabled(
            "Awake: granular %u | liquid %u | gas %u | thermal %u",
            materialTimings.activeGranularChunks,
            materialTimings.activeLiquidChunks,
            materialTimings.activeGasChunks,
            materialTimings.activeThermalChunks);
        ImGui::TextDisabled(
            "Microtiles: granular %u | liquid %u | gas %u | thermal %u",
            materialTimings.activeGranularMicrotiles,
            materialTimings.activeLiquidMicrotiles,
            materialTimings.activeGasMicrotiles,
            materialTimings.activeThermalMicrotiles);
        ImGui::TextDisabled(
            "Parallel scheduler: %u threads | granular %u | liquid %u | "
            "gas %u | thermal %u chunks",
            materialTimings.materialWorkerThreads,
            materialTimings.parallelGranularChunks,
            materialTimings.parallelLiquidChunks,
            materialTimings.parallelGasChunks,
            materialTimings.parallelThermalChunks);
        ImGui::TextDisabled(
            "Sand transfers: %u proposed | %u accepted | %u conflicts",
            materialTimings.granularMoveProposals,
            materialTimings.granularMovesAccepted,
            materialTimings.granularMoveConflicts);
        ImGui::TextDisabled(
            "Gas transfers: %u proposed | %u accepted | %u conflicts",
            materialTimings.gasMoveProposals,
            materialTimings.gasMovesAccepted,
            materialTimings.gasMoveConflicts);
        ImGui::TextDisabled(
            "Liquid transfers: %u proposed | %u accepted | %u conflicts",
            materialTimings.liquidMoveProposals,
            materialTimings.liquidMovesAccepted,
            materialTimings.liquidMoveConflicts);
        ImGui::TextDisabled(
            "Liquid prep %u cells | Edge cache %u columns | "
            "Equalization seeds %u cells",
            materialTimings.liquidPreparationCellVisits,
            materialTimings.liquidHeadSummaryHits,
            materialTimings.liquidEqualizationSeedVisits);
    } else {
        ImGui::TextDisabled("CPU material timings: collecting...");
    }
    ImGui::Separator();

    bool applyDisplay = false;
    if (ImGui::CollapsingHeader("Display",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        constexpr std::array modeNames{
            "Windowed",
            "Borderless window",
            "Borderless fullscreen",
            "Exclusive fullscreen",
        };
        const char* resolutionPreview =
            displayResolutions[static_cast<std::size_t>(
                displaySettings.resolutionIndex)]
                .label;
        if (ImGui::BeginCombo("Display resolution",
                              resolutionPreview)) {
            for (std::size_t index = 0;
                 index < displayResolutions.size(); ++index) {
                const bool selected =
                    displaySettings.resolutionIndex ==
                    static_cast<int>(index);
                if (ImGui::Selectable(displayResolutions[index].label,
                                      selected)) {
                    displaySettings.resolutionIndex =
                        static_cast<int>(index);
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::Combo("Window mode", &displaySettings.windowMode,
                     modeNames.data(),
                     static_cast<int>(modeNames.size()));
        ImGui::Text("Material grid: %d x %d",
                    gunpowder::World::viewWidth,
                    gunpowder::World::viewHeight);
        if (displaySettings.windowMode ==
            static_cast<int>(WindowMode::borderlessFullscreen)) {
            ImGui::TextDisabled(
                "Borderless fullscreen uses the desktop resolution.");
        }
        if (ImGui::Button("Apply display settings")) {
            applyDisplay = true;
        }
        if (!displayError.empty()) {
            ImGui::TextColored(ImVec4(1.0F, 0.35F, 0.30F, 1.0F),
                               "Display change failed: %s",
                               displayError.c_str());
        }
    }

    if (ImGui::CollapsingHeader(
            "Rendering profiles", ImGuiTreeNodeFlags_DefaultOpen)) {
        const char* selectedPreview =
            profiles.selected >= 0 &&
                    profiles.selected <
                        static_cast<int>(profiles.names.size())
                ? profiles.names[static_cast<std::size_t>(
                      profiles.selected)]
                      .c_str()
                : "No saved profile selected";
        ImGui::SetNextItemWidth(280.0F);
        if (ImGui::BeginCombo("Saved profiles", selectedPreview)) {
            for (std::size_t index = 0;
                 index < profiles.names.size(); ++index) {
                const bool selected =
                    profiles.selected ==
                    static_cast<int>(index);
                if (ImGui::Selectable(
                        profiles.names[index].c_str(), selected)) {
                    profiles.selected =
                        static_cast<int>(index);
                    profiles.name.fill('\0');
                    const std::string& selectedName =
                        profiles.names[index];
                    std::copy_n(
                        selectedName.begin(),
                        std::min(
                            selectedName.size(),
                            profiles.name.size() - 1),
                        profiles.name.begin());
                }
                if (selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SetNextItemWidth(280.0F);
        ImGui::InputText(
            "Profile name", profiles.name.data(),
            profiles.name.size());

        if (ImGui::Button("Save / overwrite")) {
            const std::string cleanedName =
                cleanProfileName(profiles.name.data());
            if (saveRenderProfile(
                    profiles.directory, cleanedName,
                    settings, profiles.error)) {
                refreshRenderProfiles(profiles);
                const auto saved = std::find(
                    profiles.names.begin(), profiles.names.end(),
                    cleanedName);
                if (saved != profiles.names.end()) {
                    profiles.selected = static_cast<int>(
                        std::distance(
                            profiles.names.begin(), saved));
                }
                profiles.name.fill('\0');
                std::copy_n(
                    cleanedName.begin(),
                    std::min(
                        cleanedName.size(),
                        profiles.name.size() - 1),
                    profiles.name.begin());
                profiles.status =
                    "Saved '" + cleanedName + "'.";
            }
        }
        ImGui::SameLine();
        const bool canLoad =
            profiles.selected >= 0 &&
            profiles.selected <
                static_cast<int>(profiles.names.size());
        ImGui::BeginDisabled(!canLoad);
        if (ImGui::Button("Load selected") && canLoad) {
            const std::string selectedName =
                profiles.names[static_cast<std::size_t>(
                    profiles.selected)];
            if (loadRenderProfile(
                    profiles.directory, selectedName,
                    settings, profiles.error)) {
                profiles.status =
                    "Loaded '" + selectedName + "'.";
            }
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Refresh")) {
            refreshRenderProfiles(profiles);
        }

        if (!profiles.error.empty()) {
            ImGui::TextColored(
                ImVec4(1.0F, 0.35F, 0.30F, 1.0F),
                "%s", profiles.error.c_str());
        } else if (!profiles.status.empty()) {
            ImGui::TextColored(
                ImVec4(0.48F, 0.88F, 0.58F, 1.0F),
                "%s", profiles.status.c_str());
        }
        ImGui::TextDisabled(
            "Folder: %s", profiles.directory.string().c_str());
    }

    if (ImGui::CollapsingHeader("Debug display",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        constexpr std::array debugDisplayNames{
            "Normal rendering",
            "GI result (denoised)",
            "GI ray bounces (raw)",
            "Material diffuse",
            "Material specular",
            "Material metallic",
            "Fluid metaball field",
        };
        ImGui::SetNextItemWidth(280.0F);
        ImGui::Combo(
            "Lighting view", &settings.debugDisplayMode,
            debugDisplayNames.data(),
            static_cast<int>(debugDisplayNames.size()));
        ImGui::TextDisabled(
            settings.debugDisplayMode == 2
                ? "Raw mode shows the unfiltered per-frame bounce samples."
                : (settings.debugDisplayMode == 6
                       ? "Shows raw reconstructed liquid coverage without "
                         "lighting or material shading."
                       : (settings.debugDisplayMode >= 3
                       ? "Material modes isolate the compact BRDF inputs."
                       : "Choose a diagnostic lighting or material view.")));
    }

    if (ImGui::CollapsingHeader(
            "Material shading", ImGuiTreeNodeFlags_DefaultOpen)) {
        sliderFloatWithReset(
            "Specular strength",
            &settings.materialSpecularStrength,
            0.0F, 3.0F, "%.2f",
            defaults.materialSpecularStrength);
        sliderFloatWithReset(
            "Surface normal detail",
            &settings.materialNormalDetail,
            0.0F, 2.5F, "%.2f",
            defaults.materialNormalDetail);
        ImGui::TextDisabled(
            "Marble");
        sliderFloatWithReset(
            "Marble polish", &settings.marblePolish,
            0.0F, 2.0F, "%.2f", defaults.marblePolish);
        sliderFloatWithReset(
            "Dark-vein reflectivity",
            &settings.marbleVeinReflectivity,
            0.0F, 3.0F, "%.2f",
            defaults.marbleVeinReflectivity);
        sliderFloatWithReset(
            "Reflective fleck density",
            &settings.marbleFleckDensity,
            0.0F, 0.15F, "%.3f",
            defaults.marbleFleckDensity);
        sliderFloatWithReset(
            "Fleck reflectivity",
            &settings.marbleFleckReflectivity,
            0.0F, 3.0F, "%.2f",
            defaults.marbleFleckReflectivity);
        sliderFloatWithReset(
            "Subsurface scattering",
            &settings.marbleSubsurfaceStrength,
            0.0F, 3.0F, "%.2f",
            defaults.marbleSubsurfaceStrength);
        sliderFloatWithReset(
            "Scatter distance",
            &settings.marbleScatterDistance,
            0.0F, 12.0F, "%.2f",
            defaults.marbleScatterDistance);
        ImGui::TextDisabled(
            "Material controls apply immediately and add no rays.");
    }

    if (ImGui::CollapsingHeader(
            "Fluid surfaces", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Liquid metaballs");
        ImGui::Checkbox(
            "Render water and oil as metaballs",
            &settings.liquidMetaballs);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            settings.liquidMetaballs = defaults.liquidMetaballs;
        }
        ImGui::PopID();
        if (settings.liquidMetaballs) {
            sliderFloatWithReset(
                "Metaball density",
                &settings.liquidMetaballDensity,
                0.25F, 2.5F, "%.2f",
                defaults.liquidMetaballDensity);
            sliderFloatWithReset(
                "Metaball radius",
                &settings.liquidMetaballRadius,
                0.45F, 5.0F, "%.2f",
                defaults.liquidMetaballRadius);
            sliderFloatWithReset(
                "Edge softness",
                &settings.liquidMetaballEdgeSoftness,
                0.005F, 0.30F, "%.3f",
                defaults.liquidMetaballEdgeSoftness);
            sliderFloatWithReset(
                "Flow stretching",
                &settings.liquidFlowStretch,
                0.0F, 2.5F, "%.2f",
                defaults.liquidFlowStretch);
            sliderFloatWithReset(
                "Pool flattening",
                &settings.liquidPoolFlattening,
                0.0F, 1.0F, "%.2f",
                defaults.liquidPoolFlattening);
            sliderFloatWithReset(
                "Metaball normal strength",
                &settings.liquidMetaballNormalStrength,
                0.0F, 3.0F, "%.2f",
                defaults.liquidMetaballNormalStrength);
            ImGui::TextDisabled("Liquid optics");
            sliderFloatWithReset(
                "Reflection strength",
                &settings.liquidReflectionStrength,
                0.0F, 3.0F, "%.2f",
                defaults.liquidReflectionStrength);
            sliderFloatWithReset(
                "Subsurface scattering",
                &settings.liquidSubsurfaceStrength,
                0.0F, 3.0F, "%.2f",
                defaults.liquidSubsurfaceStrength);
            sliderFloatWithReset(
                "Caustic strength",
                &settings.liquidCausticStrength,
                0.0F, 3.0F, "%.2f",
                defaults.liquidCausticStrength);
            sliderFloatWithReset(
                "Spectral dispersion",
                &settings.liquidDispersionStrength,
                0.0F, 2.0F, "%.2f",
                defaults.liquidDispersionStrength);
            ImGui::TextDisabled(
                "Density joins forms; radius changes reach; softness "
                "controls the transition.");
        }
    }

    if (ImGui::CollapsingHeader("Ray tracing",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        sliderIntWithReset(
            "Rays per light", &settings.raysPerLight, 1, 1024,
            defaults.raysPerLight, ImGuiSliderFlags_Logarithmic);
        sliderFloatWithReset(
            "Soft-shadow radius", &settings.softShadowRadius, 0.0F,
            6.0F * gunpowder::World::simulationScale, "%.2f",
            defaults.softShadowRadius);
        sliderFloatWithReset(
            "Solid transmission", &settings.solidTransmission,
            0.0F, 0.5F, "%.3f", defaults.solidTransmission);
        sliderFloatWithReset(
            "Smoke transmission", &settings.smokeTransmission,
            0.75F, 1.0F, "%.3f", defaults.smokeTransmission);
        sliderFloatWithReset(
            "Steam transmission", &settings.steamTransmission,
            0.85F, 1.0F, "%.3f", defaults.steamTransmission);
        if (settings.raysPerLight > 64) {
            ImGui::TextColored(ImVec4(1.0F, 0.72F, 0.25F, 1.0F),
                               "Experimental ray counts can make frames very slow.");
        }
        if (settings.raysPerLight > 512) {
            ImGui::TextColored(ImVec4(1.0F, 0.32F, 0.25F, 1.0F),
                               "Extreme values may trigger the GPU timeout.");
        }
    }

    if (ImGui::CollapsingHeader("Player light",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        sliderFloatWithReset(
            "Player radius", &settings.playerLightRadius,
            8.0F * gunpowder::World::simulationScale,
            160.0F * gunpowder::World::simulationScale, "%.1f",
            defaults.playerLightRadius);
        sliderFloatWithReset(
            "Player intensity", &settings.playerLightIntensity,
            0.0F, 3.0F, "%.2f", defaults.playerLightIntensity);
    }

    if (ImGui::CollapsingHeader(
            "Global illumination", ImGuiTreeNodeFlags_DefaultOpen)) {
        sliderIntWithReset(
            "Indirect bounces", &settings.indirectBounces, 0, 8,
            defaults.indirectBounces);
        sliderIntWithReset(
            "GI rays", &settings.giRays, 1, 32, defaults.giRays,
            ImGuiSliderFlags_Logarithmic);
        sliderFloatWithReset(
            "GI intensity", &settings.giIntensity, 0.0F, 2.0F,
            "%.2f", defaults.giIntensity);
        sliderFloatWithReset(
            "Bounce distance", &settings.bounceDistance,
            4.0F * gunpowder::World::simulationScale,
            128.0F * gunpowder::World::simulationScale, "%.1f",
            defaults.bounceDistance, ImGuiSliderFlags_Logarithmic);
        ImGui::PushID("Temporal GI denoising");
        ImGui::Checkbox("Temporal GI denoising",
                        &settings.temporalGiDenoising);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            settings.temporalGiDenoising =
                defaults.temporalGiDenoising;
        }
        ImGui::PopID();
        if (settings.temporalGiDenoising) {
            sliderFloatWithReset(
                "GI history stability", &settings.giHistoryWeight,
                0.0F, 0.985F, "%.3f", defaults.giHistoryWeight);
            ImGui::PushID("Adaptive GI sampling");
            ImGui::Checkbox(
                "Adaptive GI sampling",
                &settings.adaptiveGiSampling);
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset")) {
                settings.adaptiveGiSampling =
                    defaults.adaptiveGiSampling;
            }
            ImGui::PopID();
            ImGui::TextDisabled(
                "Higher stability is smoother but reacts more slowly.");
            ImGui::TextDisabled(
                "Adaptive sampling traces stable cells less often.");
        }
        if (settings.indirectBounces * settings.giRays > 64) {
            ImGui::TextColored(
                ImVec4(1.0F, 0.72F, 0.25F, 1.0F),
                "High GI sample counts can reduce performance.");
        }
    }

    if (ImGui::CollapsingHeader("Fire lights")) {
        sliderFloatWithReset(
            "Fire base radius", &settings.fireBaseRadius,
            2.0F * gunpowder::World::simulationScale,
            80.0F * gunpowder::World::simulationScale, "%.1f",
            defaults.fireBaseRadius);
        sliderFloatWithReset(
            "Cluster radius scale", &settings.fireClusterRadiusScale,
            0.0F, 12.0F, "%.2f", defaults.fireClusterRadiusScale);
        sliderFloatWithReset(
            "Maximum radius bonus", &settings.fireMaximumRadiusBonus,
            0.0F, 80.0F * gunpowder::World::simulationScale, "%.1f",
            defaults.fireMaximumRadiusBonus);
        sliderFloatWithReset(
            "Fire base intensity", &settings.fireBaseIntensity,
            0.0F, 2.0F, "%.2f", defaults.fireBaseIntensity);
        sliderFloatWithReset(
            "Cluster intensity scale",
            &settings.fireClusterIntensityScale, 0.0F, 0.05F, "%.4f",
            defaults.fireClusterIntensityScale);
        sliderFloatWithReset(
            "Maximum fire intensity", &settings.fireMaximumIntensity,
            0.0F, 3.0F, "%.2f", defaults.fireMaximumIntensity);
    }

    if (ImGui::CollapsingHeader(
            "Day, night & sky", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("Day/night cycle");
        ImGui::Checkbox("Run day/night cycle",
                        &settings.dayNightCycle);
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset")) {
            settings.dayNightCycle = defaults.dayNightCycle;
        }
        ImGui::PopID();
        sliderFloatWithReset(
            "Time of day", &settings.timeOfDayHours,
            0.0F, 24.0F, "%.2f h", defaults.timeOfDayHours);
        sliderFloatWithReset(
            "Day length", &settings.dayLengthSeconds,
            30.0F, 1200.0F, "%.0f s", defaults.dayLengthSeconds,
            ImGuiSliderFlags_Logarithmic);
        sliderIntWithReset(
            "Sun rays", &settings.sunRays, 1, 16,
            defaults.sunRays, ImGuiSliderFlags_Logarithmic);
        sliderFloatWithReset(
            "Sun intensity", &settings.sunIntensity,
            0.0F, 3.0F, "%.2f", defaults.sunIntensity);
        sliderFloatWithReset(
            "Sun shadow softness", &settings.sunShadowSoftness,
            0.0F, 0.10F, "%.3f",
            defaults.sunShadowSoftness);
        sliderFloatWithReset(
            "Sky intensity", &settings.skyIntensity,
            0.0F, 2.0F, "%.2f", defaults.skyIntensity);
        sliderFloatWithReset(
            "Sky light intensity", &settings.skyLightIntensity,
            0.0F, 1.0F, "%.2f", defaults.skyLightIntensity);
        if (settings.sunRays > 4) {
            ImGui::TextColored(
                ImVec4(1.0F, 0.72F, 0.25F, 1.0F),
                "Directional rays affect every visible material cell.");
        }
    }

    if (ImGui::CollapsingHeader("Ambient light & haze",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
        sliderFloatWithReset(
            "Night ambient", &settings.ambientIntensity,
            0.0F, 0.12F, "%.3f", defaults.ambientIntensity);
        sliderFloatWithReset(
            "Daylight ambient", &settings.daylightAmbientIntensity,
            0.0F, 0.20F, "%.3f",
            defaults.daylightAmbientIntensity);
        sliderFloatWithReset(
            "Base haze", &settings.baseHaze, 0.0F, 0.50F, "%.3f",
            defaults.baseHaze);
        sliderFloatWithReset(
            "Smoke haze contribution",
            &settings.smokeHazeContribution, 0.0F, 2.0F, "%.2f",
            defaults.smokeHazeContribution);
        sliderFloatWithReset(
            "Haze light absorption", &settings.hazeAttenuation,
            0.0F, 1.0F, "%.2f", defaults.hazeAttenuation);
    }

    ImGui::Separator();
    if (ImGui::Button("Reset rendering defaults")) {
        settings = gunpowder::RayTracingSettings{};
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Press ` to hide");
    ImGui::End();
    return applyDisplay;
}

} // namespace

int main(int, char**) {
    try {
        SdlContext sdl;
        DisplaySettings displaySettings;
        const DisplayResolution& defaultResolution =
            displayResolutions[static_cast<std::size_t>(
                displaySettings.resolutionIndex)];
        std::unique_ptr<SDL_Window, WindowDeleter> window(SDL_CreateWindow(
            "Gunpowder | A/D move, E grapple, LMB fire, RMB grenade",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            defaultResolution.width, defaultResolution.height,
            SDL_WINDOW_VULKAN | SDL_WINDOW_BORDERLESS));
        if (!window) {
            throw std::runtime_error(SDL_GetError());
        }
        std::string displayError =
            applyDisplaySettings(window.get(), displaySettings);

        gunpowder::VulkanRenderer renderer(window.get());
        gunpowder::World world;
        RenderProfileUi renderProfiles(
            renderProfileDirectory());
        refreshRenderProfiles(renderProfiles);
        world.setGpuParticlesEnabled(true);
        // Bulk materials use the local cellular model. Keeping simulation
        // CPU-resident avoids a full-grid GPU fence and readback every tick.
        world.setGpuMaterialSimulationEnabled(false);

        using Clock = std::chrono::steady_clock;
        constexpr double fixedStep = 1.0 / 120.0;
        auto previousTime = Clock::now();
        auto titleUpdateTime = previousTime;
        int framesSinceTitleUpdate = 0;
        double accumulator = 0.0;
        bool running = true;
        bool jumpHeld = false;
        bool grenadeHeld = false;
        bool jumpQueued = false;
        bool grenadeQueued = false;
        bool grappleToggleQueued = false;
        bool developerUiOpen = false;
        gunpowder::Material paintMaterial = gunpowder::Material::water;
        bool liquidDebug = false;
        float displayedFramesPerSecond = 0.0F;

        while (running) {
            SDL_Event event{};
            while (SDL_PollEvent(&event) != 0) {
                ImGui_ImplSDL2_ProcessEvent(&event);
                if (event.type == SDL_QUIT) {
                    running = false;
                } else if (event.type == SDL_KEYDOWN && !event.key.repeat) {
                    if (event.key.keysym.sym == SDLK_BACKQUOTE) {
                        developerUiOpen = !developerUiOpen;
                    } else if (event.key.keysym.sym == SDLK_ESCAPE &&
                               developerUiOpen) {
                        developerUiOpen = false;
                    } else if (event.key.keysym.sym == SDLK_ESCAPE) {
                        running = false;
                    } else if (!ImGui::GetIO().WantCaptureKeyboard &&
                               event.key.keysym.sym == SDLK_r) {
                        world.regenerate();
                    } else if (!ImGui::GetIO().WantCaptureKeyboard &&
                               event.key.keysym.sym == SDLK_F3) {
                        liquidDebug = !liquidDebug;
                    } else if (!ImGui::GetIO().WantCaptureKeyboard &&
                               event.key.keysym.sym == SDLK_F4) {
                        renderer.rayTracingSettings().debugDisplayMode =
                            renderer.rayTracingSettings().debugDisplayMode == 6
                                ? 0
                                : 6;
                    } else if (!ImGui::GetIO().WantCaptureKeyboard &&
                               event.key.keysym.sym == SDLK_f) {
                        renderer.setPlayerLightEnabled(
                            !renderer.playerLightEnabled());
                    } else if (!ImGui::GetIO().WantCaptureKeyboard &&
                               event.key.keysym.sym == SDLK_e) {
                        grappleToggleQueued = true;
                    }
                }
            }

            const auto now = Clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - previousTime).count();
            previousTime = now;
            accumulator += std::min(elapsed, 0.1);

            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            const bool keyboardCaptured =
                developerUiOpen && ImGui::GetIO().WantCaptureKeyboard;
            const bool mouseCaptured =
                developerUiOpen && ImGui::GetIO().WantCaptureMouse;
            const gunpowder::Material previousMaterial = paintMaterial;
            if (!keyboardCaptured && keys[SDL_SCANCODE_1] != 0) {
                paintMaterial = gunpowder::Material::sand;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_2] != 0) {
                paintMaterial = gunpowder::Material::water;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_3] != 0) {
                paintMaterial = gunpowder::Material::oil;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_4] != 0) {
                paintMaterial = gunpowder::Material::fire;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_5] != 0) {
                paintMaterial = gunpowder::Material::smoke;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_6] != 0) {
                paintMaterial = gunpowder::Material::wood;
            } else if (!keyboardCaptured && keys[SDL_SCANCODE_7] != 0) {
                paintMaterial = gunpowder::Material::metal;
            }
            if (paintMaterial != previousMaterial) {
                const std::string title =
                    std::string("Gunpowder | Paint: ") +
                    materialName(paintMaterial) +
                    " | MMB/P paint, LMB fire, RMB grenade";
                SDL_SetWindowTitle(window.get(), title.c_str());
            }
            int mouseX = 0;
            int mouseY = 0;
            const Uint32 mouseButtons = SDL_GetMouseState(&mouseX, &mouseY);
            int windowWidth = 1;
            int windowHeight = 1;
            SDL_GetWindowSize(window.get(), &windowWidth, &windowHeight);

            gunpowder::InputState input;
            input.moveLeft =
                !keyboardCaptured && keys[SDL_SCANCODE_A] != 0;
            input.moveRight =
                !keyboardCaptured && keys[SDL_SCANCODE_D] != 0;
            input.grappleToggle =
                !keyboardCaptured && grappleToggleQueued;
            const bool grappleAttached = world.grapple().attached;
            const bool spaceDown =
                !keyboardCaptured && keys[SDL_SCANCODE_SPACE] != 0;
            const bool jumpNow =
                spaceDown ||
                (!keyboardCaptured && !grappleAttached &&
                 keys[SDL_SCANCODE_W] != 0);
            jumpQueued = jumpQueued || (jumpNow && !jumpHeld);
            jumpHeld = jumpNow;
            // Space remains a dedicated jump/swim control while attached.
            // W changes to reel-in only after the hook has anchored.
            input.jump = jumpQueued;
            input.swimUp =
                spaceDown ||
                (!keyboardCaptured && !grappleAttached &&
                 keys[SDL_SCANCODE_W] != 0);
            input.reelIn =
                !keyboardCaptured && grappleAttached &&
                (keys[SDL_SCANCODE_W] != 0 ||
                 keys[SDL_SCANCODE_UP] != 0);
            input.reelOut =
                !keyboardCaptured && grappleAttached &&
                (keys[SDL_SCANCODE_S] != 0 ||
                 keys[SDL_SCANCODE_DOWN] != 0);
            input.fire =
                !mouseCaptured &&
                (mouseButtons & SDL_BUTTON(SDL_BUTTON_LEFT)) != 0U;
            const bool grenadeNow =
                (!mouseCaptured &&
                 (mouseButtons & SDL_BUTTON(SDL_BUTTON_RIGHT)) != 0U) ||
                (!keyboardCaptured && keys[SDL_SCANCODE_G] != 0);
            grenadeQueued = grenadeQueued || (grenadeNow && !grenadeHeld);
            grenadeHeld = grenadeNow;
            input.throwGrenade = grenadeQueued;
            input.paint =
                (!mouseCaptured &&
                 (mouseButtons & SDL_BUTTON(SDL_BUTTON_MIDDLE)) != 0U) ||
                (!keyboardCaptured && keys[SDL_SCANCODE_P] != 0);
            input.paintMaterial = paintMaterial;
            input.liquidDebug = liquidDebug;
            const gunpowder::Vec2 cameraTopLeft = world.cameraTopLeft();
            input.aim = cameraTopLeft + gunpowder::Vec2{
                static_cast<float>(mouseX) /
                    static_cast<float>(std::max(windowWidth, 1)) *
                    static_cast<float>(gunpowder::World::viewWidth),
                static_cast<float>(mouseY) /
                    static_cast<float>(std::max(windowHeight, 1)) *
                    static_cast<float>(gunpowder::World::viewHeight),
            };

            while (accumulator >= fixedStep) {
                world.update(static_cast<float>(fixedStep), input);
                // Edge-triggered actions only apply to the first catch-up tick.
                input.jump = false;
                input.throwGrenade = false;
                input.grappleToggle = false;
                jumpQueued = false;
                grenadeQueued = false;
                grappleToggleQueued = false;
                accumulator -= fixedStep;
            }
            renderer.beginUiFrame();
            bool applyDisplay = false;
            if (developerUiOpen) {
                applyDisplay = drawDeveloperUi(
                    &developerUiOpen, renderer.rayTracingSettings(),
                    renderer.gpuRayTimings(),
                    world.materialSimulationTimings(),
                    displaySettings, renderProfiles,
                    displayError,
                    displayedFramesPerSecond);
            }
            ImGui::Render();
            if (applyDisplay) {
                renderer.waitIdle();
                displayError =
                    applyDisplaySettings(window.get(), displaySettings);
                if (displayError.empty()) {
                    renderer.refreshDisplay();
                }
            }
            renderer.draw(world);
            ++framesSinceTitleUpdate;
            const double titleElapsed =
                std::chrono::duration<double>(now - titleUpdateTime).count();
            if (titleElapsed >= 0.5) {
                const int framesPerSecond = static_cast<int>(
                    static_cast<double>(framesSinceTitleUpdate) /
                        titleElapsed +
                    0.5);
                displayedFramesPerSecond =
                    static_cast<float>(framesPerSecond);
                const gunpowder::Player& player = world.player();
                std::string status;
                if (player.wetness > 0.12F) {
                    status += " WET";
                }
                if (player.oiliness > 0.12F) {
                    status += " OILY";
                }
                if (player.burning > 0.08F) {
                    status += " BURNING";
                }
                if (player.suffocation > 0.35F) {
                    status += " SMOKE";
                }
                const std::string title =
                    std::string("Gunpowder | ") +
                    std::to_string(framesPerSecond) +
                    " FPS | HP " +
                    std::to_string(static_cast<int>(player.health + 0.5F)) +
                    status +
                    (world.grapple().attached ? " | GRAPPLED" : "") +
                    (!renderer.playerLightEnabled()
                         ? " | LIGHT OFF"
                         : "") +
                    " | Paint: " + materialName(paintMaterial) +
                    (liquidDebug ? " | Flow debug" : "");
                SDL_SetWindowTitle(window.get(), title.c_str());
                titleUpdateTime = now;
                framesSinceTitleUpdate = 0;
            }
        }
        renderer.waitIdle();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fatal error: " << error.what() << '\n';
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Gunpowder failed",
                                 error.what(), nullptr);
        return 1;
    }
}
