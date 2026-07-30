# Gunpowder

A small C++ prototype for a Noita-inspired action game built around conventional
guns, grenades, and destructible terrain.

The current milestone includes:

- SDL2 windowing and input
- a Vulkan 1.1 material-texture renderer
- portable 2D ray-traced lighting with soft terrain shadows
- a procedural day/night sky with a directional ray-cast sun
- smoke-responsive volumetric haze and ray-traced grapple lighting
- compact diffuse/specular/metallic material shading with GGX highlights and
  thickness-aware marble subsurface scattering
- flow-aware metaball skins for water and oil, with gradient-derived normals,
  Fresnel reflections, depth absorption, subsurface color, and caustics
- named, human-readable rendering profiles for lighting and material looks
- visible-region texture uploads instead of full-world transfers
- fixed-timestep gameplay
- a 1024 x 576 design-space world sampled by a higher-resolution material grid
- a smooth player-following camera
- chunk-aligned simulation around the visible area
- cell-based destructible terrain with falling sand
- a stable tomb entrance followed by seeded modular burial rooms, flooded
  chambers, oil vaults, collapses, and wooden grave-robber staging
- destructible wood that catches fire and spreads flames through structures
- discrete full-cell water/oil flow with material-specific mobility
- spreading fire, rising smoke, heat transfer, evaporation, and condensation
- swimming, buoyancy, liquid drag, displacement, and splash particles
- wetness, oil coating, burning, smoke exposure, health, and respawning
- liquid-aware grenade buoyancy and projectile drag
- player movement and jumping
- rounded swept player collision with corner sliding
- an unbreakable physics grappling hook that swings and wraps around terrain
- material-aware penetrating bullets
- bouncing grenades that displace liquids and ignite oil
- explosion particles, flashes, and screen shake
- live FPS reporting in the window title

## Build

Prerequisites:

- CMake 3.25+
- a C++20 compiler
- SDL2 development files
- the Vulkan SDK (including `glslc`)

With the MSYS2 UCRT64 toolchain used during development:

```powershell
cmake -S . -B build-ucrt -G "MinGW Makefiles" `
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/clang++.exe `
  -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/mingw32-make.exe
cmake --build build-ucrt --parallel
./build-ucrt/gunpowder.exe
```

On other platforms, point CMake at an SDL2 package and use the normal compiler
configuration for that environment.

## Controls

| Input | Action |
| --- | --- |
| A / D | Move |
| W / Space | Jump or swim upward; Space still jumps while grappled |
| Tap E | Fire the grappling hook; tap E again to detach |
| W / Up while attached | Reel the grappling chain in |
| S / Down while attached | Reel the grappling chain out |
| Left mouse | Fire |
| Right mouse / G | Throw grenade |
| 1-7 | Select sand, water, oil, fire, smoke, wood, or metal |
| Middle mouse / P | Paint the selected material at the cursor |
| F3 | Toggle the liquid volume/flow diagnostic overlay |
| F4 | Toggle the raw normalized metaball-field view |
| ` | Toggle developer settings, fluid look controls, and rendering profiles |
| R | Regenerate the procedural world |
| Escape | Quit |

## Day, night, and sky

The world now runs through a configurable 24-hour cycle. The procedural sky
changes from night through sunrise, daylight, sunset, and back to night, with
stars appearing after dark. The sun is a directional ray-cast light rather
than a screen-space glow: terrain, wood, smoke, and steam affect its rays, and
sunlight also contributes to haze and global illumination.

The developer settings can pause the cycle and scrub the time of day, change
the day length, sky display and sky-light intensity, sun intensity, shadow
softness, and sun-ray count. Ambient light interpolates from a configurable
night floor to a brighter daytime value along the daylight curve. The sky is
also a cool diffuse light source for vertically exposed air, roof surfaces,
outside walls, and open liquid surfaces; unlike the directional sun, it does
not create a second hard shadow. The default cycle lasts four minutes and uses
one sun ray per cell; extra rays soften the directional shadows at an
additional rendering cost. Sky exposure is cached across the full world and
refreshed when terrain changes, so roofs continue to block the sky even when
they are outside the current view. Tomb interiors also retain a separate
persistent backdrop: destroying a wall or roof can admit ray-cast sunlight
and diffuse skylight without replacing the room itself with the outdoor sky.
Directional sun blockers are projected into a four-samples-per-
cell horizon spanning the complete material world. Rasterizing each solid's
full projected footprint prevents diagonal grid gaps. Each quantized horizon
ray stores the exact exit depth of its foremost sun-facing solid rather than a
single approximate depth for the whole cell, avoiding shallow-angle bands and
roof self-shadowing. Buried solid cells are skipped because they cannot change
the silhouette. The visible compute trace adds smoke and steam attenuation,
but solid occlusion remains entirely world-space. There is no viewport-
dependent handoff, so structures beyond the viewport cast stable shadows that
do not follow the camera.

## Material simulation

Bulk materials use a Noita-style local cellular model. Each liquid cell is a
full water or oil pixel: it falls, tries downward diagonals, then searches
sideways for open space or a nearby drop. Water has a longer lateral search
than viscous oil, and locally displaces oil downward so oil rises to the
surface. Liquid cells are gathered directly from awake chunks and processed
bottom-up, with randomized same-row order and one-move source/destination
locks. Narrow streams can follow freshly vacated cells without alternating
gaps, while diagonal free-surface detection and longer lateral relocation keep
pools from freezing into stair-stepped mounds. Active columns cache the depth
of each connected liquid run, giving boundary cells a cheap hydrostatic-head
signal: deep water pushes farther through openings and chooses the lower of
nearby outlets, while oil uses shorter reach and a higher pressure threshold.
Only surfaces, outlets, density boundaries, and moving cells enter the liquid
worklist. Lateral momentum persists between moves instead of being recreated
randomly every tick. A conservative connected-body pass also compares distant
supported free surfaces that share a submerged or diagonal connection, moving
existing full cells from the highest surface to the lowest until communicating
reservoirs reach the same level. Those transfers are planned once, reserved,
and distributed between the ordinary liquid substeps, so pressure equalization
appears as part of the ongoing flow rather than as a separate correction.
When a connected body has no downward path, impact foam, density inversion, or
surface difference greater than one cell, its remaining directional momentum
is cleared and the body sleeps. Later terrain or entity disturbances wake it
normally, preventing otherwise stable top rows from shuffling forever.

Fast water impacts leave short-lived foam metadata and emit sparse spray
particles. The material shader uses packed flow direction for moving highlights
and foam breakup without changing or duplicating simulated liquid cells. In
normal rendering, water and oil receive a separate 3x3 implicit metaball skin.
Resting fields spread horizontally, moving fields stretch with flow, and the
analytic field gradient supplies a smooth reflection normal. The developer
window exposes density, radius, edge softness, flow stretching, pool
flattening, normal strength, reflection, subsurface scattering, and caustics;
the cellular simulation and conserved volume remain unchanged. The same
controls expose polished-marble scattering strength and distance. Marble
estimates local thickness from nearby samples, while liquids estimate depth
for wavelength-dependent absorption. These optical effects reuse the existing
direct, sky, and GI results and add no ray-tracing passes.
Smoke, steam, fire, and sand use corresponding local movement and reaction
rules.

The developer window can save and load named rendering profiles. Profiles cover
ray tracing, GI, day/night lighting, atmosphere, material shading, marble, and
fluid-surface controls while deliberately leaving window/display mode alone.
They are versioned text files in the platform user-preferences directory under
`Gunpowder/render-profiles`, so looks can be archived, compared, or edited
outside the game.

At the default 1920 x 1080 display, the visible material field is 960 x 540:
each simulated material pixel occupies 2 x 2 display pixels. Rendering and
display resolution remain independent from the material grid.

The simulation is divided into 64x64 activity chunks. A cell change wakes its
chunk and neighboring chunks; stable regions sleep after several unchanged
ticks. Splashes and other fast fragments temporarily use the separate velocity
particle system before visual dissipation. Bulk material simulation remains
CPU-resident, avoiding per-tick GPU readback and synchronization.

## Direction

The prototype deliberately keeps `World` independent from Vulkan. The next
useful milestone is a complete combat loop: a first enemy, ammo and reloading,
weapon definitions, pickups, and a more expressive HUD.
