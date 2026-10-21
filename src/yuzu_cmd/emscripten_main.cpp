// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// WebAssembly frontend. main() runs on a worker thread (PROXY_TO_PTHREAD) and owns the emulated
// system; the exported C functions only queue requests or read atomics, so they are safe to call
// from the browser's main thread without blocking it.
//
// Built for Node.js (EDEN_WEB_HEADLESS), main() boots the ROM given on the command line and
// reports performance until the requested number of seconds has elapsed.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include <emscripten.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/cpu_manager.h"
#include "core/crypto/key_manager.h"
#include "core/file_sys/registered_cache.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/service/am/applet_manager.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/arm/arm_interface.h"
#include "core/memory.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/perf_stats.h"
#include "hid_core/hid_core.h"
#include "input_common/drivers/virtual_gamepad.h"
#include "input_common/main.h"
#include "video_core/gpu.h"

namespace {

/// Window without a native surface; the web renderer presents to the page canvas itself.
class WebEmuWindow final : public Core::Frontend::EmuWindow {
public:
    WebEmuWindow() {
        window_info.type = Core::Frontend::WindowSystemType::Headless;
        UpdateCurrentFramebufferLayout(Layout::ScreenUndocked::Width, Layout::ScreenUndocked::Height);
    }

    std::unique_ptr<Core::Frontend::GraphicsContext> CreateSharedContext() const override {
        return std::make_unique<Core::Frontend::GraphicsContext>();
    }

    bool IsShown() const override {
        return true;
    }
};

enum class EmuState : int {
    Idle = 0,
    Booting = 1,
    Running = 2,
    Paused = 3,
    Failed = 4,
    Stopped = 5,
};

struct Request {
    enum class Kind { Boot, Pause, Resume, Shutdown, Button, Stick } kind;
    std::string path;
    /// Input: player index, VirtualGamepad button or stick id, and its state.
    int player = 0;
    int id = 0;
    bool pressed = false;
    float x = 0.0f;
    float y = 0.0f;
};

std::mutex g_request_mutex;
std::condition_variable g_request_cv;
std::deque<Request> g_requests;

std::atomic<int> g_state{int(EmuState::Idle)};
std::atomic<int> g_load_result{0};
std::atomic<double> g_game_fps{0.0};
std::atomic<double> g_system_fps{0.0};
std::atomic<double> g_emulation_speed{0.0};
std::atomic<double> g_frametime_ms{0.0};

void PushRequest(Request request) {
    {
        std::scoped_lock lock{g_request_mutex};
        g_requests.push_back(std::move(request));
    }
    g_request_cv.notify_one();
}

void ApplyWebSettings() {
    // The Switch exposes four CPU cores; run one host thread per emulated core.
    Settings::values.use_multi_core.SetValue(true);
    Settings::values.cpu_backend.SetValue(Settings::CpuBackend::Dynarmic);
#ifdef EDEN_WEB_HEADLESS
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Null);
#else
    // Maps to the WebGPU renderer on the web.
    Settings::values.renderer_backend.SetValue(Settings::RendererBackend::Vulkan);
#endif
    Settings::values.use_asynchronous_gpu_emulation.SetValue(true);
    Settings::values.use_disk_shader_cache.SetValue(false);
    Settings::values.sink_id.SetValue(Settings::AudioEngine::Null);
}

class Emulator {
public:
    bool Boot(const std::string& path) {
        g_state = int(EmuState::Booting);
        LOG_INFO(Frontend, "Booting {}", path);
        // The page picks the CPU core by setting EDEN_INTERP_JIT before the runtime starts.
        const char* jit = std::getenv("EDEN_INTERP_JIT");
        LOG_INFO(Frontend, "CPU core: {}", jit && jit[0] == '0' ? "IR interpreter" : "Wasm JIT");

        // Keys and firmware are mounted by the page after the runtime starts.
        Core::Crypto::KeyManager::Instance().ReloadKeys();

        system.SetContentProvider(std::make_unique<FileSys::ContentProviderUnion>());
        system.SetFilesystem(std::make_shared<FileSys::RealVfsFilesystem>());
        system.GetFileSystemController().CreateFactories(*system.GetFilesystem());
        system.GetUserChannel().clear();

        Service::AM::FrontendAppletParameters params{
            .applet_id = Service::AM::AppletId::Application,
        };
        const Core::SystemResultStatus result = system.Load(window, path, params);
        g_load_result = int(result);
        if (result != Core::SystemResultStatus::Success) {
            LOG_CRITICAL(Frontend, "Failed to load {} (result {})", path, u32(result));
            g_state = int(EmuState::Failed);
            return false;
        }

        // Controllers bind to their input devices when the application is loaded.
        system.HIDCore().ReloadInputDevices();
        system.GPU().Start();
        system.GetCpuManager().OnGpuReady();
        system.RegisterExitCallback([] { PushRequest({Request::Kind::Shutdown, {}}); });

        system.Run();
        g_state = int(EmuState::Running);
        return true;
    }

    void Pause() {
        if (g_state == int(EmuState::Running)) {
            void(system.Pause());
            g_state = int(EmuState::Paused);
        }
    }

    void Resume() {
        if (g_state == int(EmuState::Paused)) {
            system.Run();
            g_state = int(EmuState::Running);
        }
    }

    void Shutdown() {
        if (g_state == int(EmuState::Running) || g_state == int(EmuState::Paused)) {
            void(system.Pause());
            system.ShutdownMainProcess();
        }
        g_state = int(EmuState::Stopped);
    }

    /// Debug aid for the headless runner: prints the application's guest threads.
    void DumpThreads() {
        auto* process = system.ApplicationProcess();
        if (!process) {
            return;
        }
        for (size_t core = 0; core < 4; ++core) {
            const auto& scheduler = system.Kernel().Scheduler(core);
            const auto* current = scheduler.GetSchedulerCurrentThread();
            LOG_INFO(Frontend, "core {} running thread {} (idle={})", core,
                     current ? current->GetThreadId() : 0, current == scheduler.GetIdleThread());
        }
        if (auto* arm = process->GetArmInterface(0)) {
            Kernel::Svc::ThreadContext live{};
            arm->GetContext(live);
            LOG_INFO(Frontend, "core 0 live pc={:#x} lr={:#x} sp={:#x} r0={:#x} r1={:#x} r2={:#x} r3={:#x} r4={:#x} r5={:#x}",
                     live.pc, live.lr, live.sp, live.r[0], live.r[1], live.r[2], live.r[3], live.r[4], live.r[5]);
            std::string stack;
            auto& memory = system.ApplicationMemory();
            for (u64 i = 0; i < 96 && memory.IsValidVirtualAddress(live.sp + i * 4); ++i) {
                const u32 word = memory.Read32(live.sp + i * 4);
                // Likely return addresses into the loaded modules.
                if (word >= 0x200000 && word < 0x8000000 && (word & 1) == 0) {
                    stack += fmt::format(" {:x}", word);
                }
            }
            LOG_INFO(Frontend, "core 0 stack scan:{}", stack);
        }
        for (const auto& thread : process->GetThreadList()) {
            const auto& context = thread.GetContext();
            LOG_INFO(Frontend, "thread {:3} state={:#x} core={} prio={} wait={} affinity={:#x} pc={:#x} lr={:#x}",
                     thread.GetThreadId(), u32(thread.GetRawState()), thread.GetActiveCore(),
                     thread.GetPriority(), u32(thread.GetWaitReasonForDebugging()),
                     thread.GetAffinityMask().GetAffinityMask(), context.pc, context.lr);
        }
    }

    void SetButton(int player, int button, bool pressed) {
        input_subsystem.GetVirtualGamepad()->SetButtonState(static_cast<size_t>(player), button,
                                                            pressed);
    }

    void SetStick(int player, int stick, float x, float y) {
        input_subsystem.GetVirtualGamepad()->SetStickPosition(static_cast<size_t>(player), stick,
                                                              x, y);
    }

    void UpdatePerfStats() {
        if (g_state != int(EmuState::Running)) {
            return;
        }
        // Stats average over the time between calls; input requests wake the loop far more often.
        const auto now = std::chrono::steady_clock::now();
        if (now - last_stats < std::chrono::seconds(1)) {
            return;
        }
        last_stats = now;
        const Core::PerfStatsResults stats = system.GetAndResetPerfStats();
        g_game_fps = stats.average_game_fps;
        g_system_fps = stats.system_fps;
        g_emulation_speed = stats.emulation_speed;
        g_frametime_ms = stats.frametime * 1000.0;
    }

    Core::System system;
    WebEmuWindow window;
    InputCommon::InputSubsystem input_subsystem;
    std::chrono::steady_clock::time_point last_stats{};
};

} // Anonymous namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE int eden_boot(const char* path) {
    if (!path || !path[0]) {
        return -1;
    }
    PushRequest({Request::Kind::Boot, path});
    return 0;
}

EMSCRIPTEN_KEEPALIVE void eden_pause() {
    PushRequest({Request::Kind::Pause, {}});
}

EMSCRIPTEN_KEEPALIVE void eden_resume() {
    PushRequest({Request::Kind::Resume, {}});
}

EMSCRIPTEN_KEEPALIVE void eden_shutdown() {
    PushRequest({Request::Kind::Shutdown, {}});
}

/// Presses or releases a controller button. `button` is a VirtualGamepad::VirtualButton
/// (0 A, 1 B, 2 X, 3 Y, 4 L stick, 5 R stick, 6 L, 7 R, 8 ZL, 9 ZR, 10 +, 11 -, 12 left,
/// 13 up, 14 right, 15 down, 16 SL, 17 SR, 18 home, 19 capture).
EMSCRIPTEN_KEEPALIVE void eden_set_button(int player, int button, int pressed) {
    PushRequest({.kind = Request::Kind::Button, .player = player, .id = button,
                 .pressed = pressed != 0});
}

/// Moves a stick (0 left, 1 right) to x, y in [-1, 1], with +y up.
EMSCRIPTEN_KEEPALIVE void eden_set_stick(int player, int stick, float x, float y) {
    PushRequest({.kind = Request::Kind::Stick, .player = player, .id = stick, .x = x, .y = y});
}

/// 0 idle, 1 booting, 2 running, 3 paused, 4 failed, 5 stopped.
EMSCRIPTEN_KEEPALIVE int eden_get_state() {
    return g_state;
}

EMSCRIPTEN_KEEPALIVE int eden_get_load_result() {
    return g_load_result;
}

EMSCRIPTEN_KEEPALIVE double eden_get_game_fps() {
    return g_game_fps;
}

EMSCRIPTEN_KEEPALIVE double eden_get_system_fps() {
    return g_system_fps;
}

EMSCRIPTEN_KEEPALIVE double eden_get_emulation_speed() {
    return g_emulation_speed;
}

EMSCRIPTEN_KEEPALIVE double eden_get_frametime() {
    return g_frametime_ms;
}

} // extern "C"

int main(int argc, char** argv) {
    Common::Log::Initialize();
    Common::Log::SetColorConsoleBackendEnabled(true);
    Common::Log::Start();
    LOG_INFO(Frontend, "Eden WebAssembly {} | {}-{}", Common::g_build_fullname,
             Common::g_scm_branch, Common::g_scm_desc);

    auto emulator = std::make_unique<Emulator>();
    ApplyWebSettings();
    emulator->system.Initialize();
    emulator->system.ApplySettings();
    // Registers the input engines, including the virtual gamepad the page drives.
    emulator->input_subsystem.Initialize();

#ifdef EDEN_WEB_HEADLESS
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <rom> [seconds] [--stats]\n", argv[0]);
        return 1;
    }
    const int seconds = argc > 2 ? std::atoi(argv[2]) : 60;
    for (int i = 3; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--stats") {
            setenv("EDEN_INTERP_STATS", "1", 1);
        } else if (arg.starts_with("--log=")) {
            Common::Log::Filter filter;
            filter.ParseFilterString(arg.substr(6));
            Common::Log::SetGlobalFilter(filter);
        } else if (arg == "--threads") {
            setenv("EDEN_DUMP_THREADS", "1", 1);
        } else if (arg == "--no-jit") {
            setenv("EDEN_INTERP_JIT", "0", 1);
        } else if (arg.starts_with("--dump-hot")) {
            // --dump-hot[=N]: dump the hottest blocks after N stats reports.
            setenv("EDEN_INTERP_DUMP_HOT", arg.size() > 11 ? std::string{arg.substr(11)}.c_str() : "0", 1);
        } else if (arg.starts_with("--dump=")) {
            setenv("EDEN_INTERP_DUMP", std::string{arg.substr(7)}.c_str(), 1);
        }
    }
    PushRequest({Request::Kind::Boot, argv[1]});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
#endif

    for (;;) {
        std::optional<Request> request;
        {
            std::unique_lock lock{g_request_mutex};
            g_request_cv.wait_for(lock, std::chrono::seconds(1), [] { return !g_requests.empty(); });
            if (!g_requests.empty()) {
                request = std::move(g_requests.front());
                g_requests.pop_front();
            }
        }

        if (request) {
            switch (request->kind) {
            case Request::Kind::Boot:
                emulator->Boot(request->path);
                break;
            case Request::Kind::Pause:
                emulator->Pause();
                break;
            case Request::Kind::Resume:
                emulator->Resume();
                break;
            case Request::Kind::Shutdown:
                emulator->Shutdown();
                break;
            case Request::Kind::Button:
                emulator->SetButton(request->player, request->id, request->pressed);
                break;
            case Request::Kind::Stick:
                emulator->SetStick(request->player, request->id, request->x, request->y);
                break;
            }
        }

        emulator->UpdatePerfStats();

#ifdef EDEN_WEB_HEADLESS
        static int dump_tick = 0;
        if (g_state == int(EmuState::Running) && std::getenv("EDEN_DUMP_THREADS") && ++dump_tick % 10 == 0) {
            emulator->DumpThreads();
        }
        if (g_state == int(EmuState::Running)) {
            LOG_INFO(Frontend, "perf: game {:.1f} fps, system {:.1f} fps, speed {:.0f}%",
                     g_game_fps.load(), g_system_fps.load(), g_emulation_speed.load() * 100.0);
        }
        if (g_state == int(EmuState::Failed) || g_state == int(EmuState::Stopped) ||
            std::chrono::steady_clock::now() > deadline) {
            break;
        }
#endif
    }

#ifdef EDEN_WEB_HEADLESS
    emulator->Shutdown();
    Common::Log::Stop();
    return g_state == int(EmuState::Failed) ? 2 : 0;
#else
    return 0;
#endif
}

#define VMA_IMPLEMENTATION
#include "video_core/vulkan_common/vma.h"
