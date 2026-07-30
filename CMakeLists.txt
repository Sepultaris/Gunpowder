cmake_minimum_required(VERSION 3.25)

project(Gunpowder
    VERSION 0.1.0
    DESCRIPTION "A destructible-terrain action prototype"
    LANGUAGES CXX
)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Build type for single-configuration generators" FORCE
    )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)
find_package(SDL2 CONFIG REQUIRED)

add_library(imgui STATIC
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_sdl2.cpp
    external/imgui/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    external/imgui
    external/imgui/backends
)
target_link_libraries(imgui PUBLIC SDL2::SDL2 Vulkan::Vulkan)

add_executable(gunpowder
    src/main.cpp
    src/game/world.cpp
    src/render/vulkan_renderer.cpp
)

target_include_directories(gunpowder PRIVATE src)
if(TARGET SDL2::SDL2main)
    target_link_libraries(gunpowder PRIVATE
        SDL2::SDL2main SDL2::SDL2 Vulkan::Vulkan imgui
    )
else()
    target_link_libraries(gunpowder PRIVATE SDL2::SDL2 Vulkan::Vulkan imgui)
endif()

if(MSVC)
    target_compile_options(gunpowder PRIVATE /W4 /permissive-)
else()
    target_compile_options(gunpowder PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS "$ENV{VULKAN_SDK}/Bin"
    REQUIRED
)

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
set(SHADER_OUTPUTS)
foreach(SHADER IN ITEMS
        world.vert world.frag cells.vert cells.frag
        lighting.comp derived_fields.comp particles.comp particles.vert
        materials.comp)
    set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/assets/shaders/${SHADER}")
    set(OUTPUT "${SHADER_OUTPUT_DIR}/${SHADER}.spv")
    add_custom_command(
        OUTPUT "${OUTPUT}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${SHADER_OUTPUT_DIR}"
        COMMAND "${GLSLC_EXECUTABLE}" "${SOURCE}" -o "${OUTPUT}"
        DEPENDS "${SOURCE}"
        VERBATIM
    )
    list(APPEND SHADER_OUTPUTS "${OUTPUT}")
endforeach()

add_custom_target(gunpowder_shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(gunpowder gunpowder_shaders)

add_custom_command(TARGET gunpowder POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${SHADER_OUTPUT_DIR}"
        "$<TARGET_FILE_DIR:gunpowder>/shaders"
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "$<TARGET_FILE_DIR:gunpowder>/sprites"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/sprites/player_hooded.bmp"
        "$<TARGET_FILE_DIR:gunpowder>/sprites/player_hooded.bmp"
)

if(WIN32 AND TARGET SDL2::SDL2)
    add_custom_command(TARGET gunpowder POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:SDL2::SDL2>"
            "$<TARGET_FILE_DIR:gunpowder>"
        COMMAND_EXPAND_LISTS
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(gunpowder_world_tests
        tests/world_simulation_tests.cpp
        src/game/world.cpp
    )
    target_include_directories(gunpowder_world_tests PRIVATE src)
    target_compile_definitions(gunpowder_world_tests PRIVATE
        GUNPOWDER_TEST_SCALE=1
    )
    if(MSVC)
        target_compile_options(gunpowder_world_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(gunpowder_world_tests PRIVATE
            -Wall -Wextra -Wpedantic
        )
    endif()
    add_test(NAME world_liquid_conservation COMMAND gunpowder_world_tests)
endif()
