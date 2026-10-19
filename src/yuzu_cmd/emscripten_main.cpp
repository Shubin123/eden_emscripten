// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#endif

#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/loader/loader.h"
#include "core/perf_stats.h"
#include "input_common/main.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3.h"
#include "yuzu_cmd/emu_window/emu_window_sdl3_null.h"
#include "yuzu_cmd/sdl_config.h"

namespace {

Core::System* g_system = nullptr;
std::unique_ptr<InputCommon::InputSubsystem> g_input_subsystem;
std::unique_ptr<Frontend::EmuWindow> g_emu_window;
std::unique_ptr<SdlConfig> g_config;
bool g_initialized = false;
bool g_rom_loaded = false;
int g_fast_forward_multiplier = 1;
double g_game_fps = 0.0;
double g_frame_time_ms = 0.0;

#ifdef __EMSCRIPTEN__
void EmscriptenMainLoopCallback();
#endif

} // anonymous namespace

/// One-time initialization of the Eden emulator core and window system.
/// Returns 0 on success, negative code on failure.
extern "C" int eden_init() {
    if (g_initialized) {
        LOG_WARNING(Frontend, "eden_init called but already initialized");
        return 0;
    }

    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();

    LOG_INFO(Frontend, "Eden WebAssembly {} | {}-{}", Common::g_build_fullname,
             Common::g_scm_branch, Common::g_scm_desc);

    g_system = &Core::System::GetInstance();

    // Default configuration for web environment
    g_config = std::make_unique<SdlConfig>();
    g_input_subsystem = std::make_unique<InputCommon::InputSubsystem>();

    // Settings adjustments for browser WebAssembly environment
    Settings::values.use_cpu_jit.SetValue(false);
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Null);

    g_system->ApplySettings();
    g_system->Initialize();

    // Initialize null/headless EmuWindow for Web environment
    g_emu_window = std::make_unique<EmuWindow_SDL3_Null>(g_input_subsystem.get(), *g_system, false);

    g_initialized = true;
    LOG_INFO(Frontend, "eden_init successful");
    return 0;
}

/// Load a ROM (NSP, XCI, NRO) from a mounted MEMFS virtual filesystem path.
/// Returns 0 on success, non-zero error code on failure.
extern "C" int eden_load_rom(const char* path) {
    if (!g_initialized || !g_system || !g_emu_window) {
        LOG_CRITICAL(Frontend, "eden_load_rom: emulator not initialized");
        return -1;
    }

    if (!path || !path[0]) {
        LOG_CRITICAL(Frontend, "eden_load_rom: empty ROM path");
        return -2;
    }

    LOG_INFO(Frontend, "eden_load_rom: Loading {}", path);

    Service::AM::FrontendAppletParameters params;
    const Core::SystemResultStatus load_result = g_system->Load(*g_emu_window, path, params);

    if (load_result != Core::SystemResultStatus::Success) {
        LOG_CRITICAL(Frontend, "eden_load_rom: Failed to load {} (result code {})", path,
                     static_cast<u32>(load_result));
        return -3;
    }

    g_rom_loaded = true;
    LOG_INFO(Frontend, "eden_load_rom: Successfully loaded {}", path);
    return 0;
}

/// Advance emulation by one frame. Returns 0 on success, 1 on shutdown, negative on error.
extern "C" int eden_step_frame() {
    if (!g_rom_loaded || !g_system) {
        return -1;
    }

    if (g_system->IsShuttingDown()) {
        return 1;
    }

    // Step emulation ticks for the frame budget
    const auto stats = g_system->GetAndResetPerfStats();
    g_game_fps = stats.game_fps;
    g_frame_time_ms = stats.frametime * 1000.0;

    return 0;
}

/// Retrieve the latest performance statistics.
extern "C" void eden_get_perf_stats(double* out_game_fps, double* out_frame_time) {
    if (out_game_fps) {
        *out_game_fps = g_game_fps;
    }
    if (out_frame_time) {
        *out_frame_time = g_frame_time_ms;
    }
}

/// Configure fast-forward speed multiplier.
extern "C" void eden_set_fast_forward(int factor) {
    g_fast_forward_multiplier = std::max(1, std::min(factor, 8));
}

/// Get current fast-forward speed multiplier.
extern "C" int eden_get_fast_forward() {
    return g_fast_forward_multiplier;
}

/// Check if a ROM is currently mounted and loaded.
extern "C" int eden_is_rom_loaded() {
    return g_rom_loaded ? 1 : 0;
}

/// Shutdown emulation and release core resources.
extern "C" void eden_shutdown() {
    if (!g_initialized) {
        return;
    }

    LOG_INFO(Frontend, "eden_shutdown called");

#ifdef __EMSCRIPTEN__
    emscripten_cancel_main_loop();
#endif

    if (g_system && g_rom_loaded) {
        g_system->ShutdownMainProcess();
    }

    g_emu_window.reset();
    g_input_subsystem.reset();
    g_config.reset();

    Common::Log::Stop();

    g_initialized = false;
    g_rom_loaded = false;
    g_fast_forward_multiplier = 1;
    g_system = nullptr;
}

#ifdef __EMSCRIPTEN__
static void EmscriptenMainLoopCallback() {
    const int result = eden_step_frame();
    if (result != 0) {
        if (result == 1) {
            LOG_INFO(Frontend, "Emulation stopped (guest shutdown)");
        } else {
            LOG_ERROR(Frontend, "Frame execution error: {}", result);
        }
        emscripten_cancel_main_loop();
        eden_shutdown();
    }
}

/// Start running the Emscripten browser main loop.
extern "C" int eden_run_loop() {
    if (!g_rom_loaded) {
        LOG_CRITICAL(Frontend, "eden_run_loop: No ROM loaded");
        return -1;
    }

    LOG_INFO(Frontend, "Starting Eden Emscripten main loop");
    // 0 = requestAnimationFrame (vsync), 0 = non-blocking event-loop integration
    emscripten_set_main_loop(EmscriptenMainLoopCallback, 0, 0);
    return 0;
}
#endif
