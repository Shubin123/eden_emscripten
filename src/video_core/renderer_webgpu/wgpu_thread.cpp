// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <exception>
#include <cmath>
#include <vector>

#include <emscripten.h>
#include <emscripten/atomic.h>
#include <emscripten/eventloop.h>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>

#include "common/assert.h"
#include "common/logging.h"
#include "common/thread.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"

extern "C" {
// Implemented in eden_webgpu.js.
void eden_webgpu_install_canvas_listener();
int eden_webgpu_canvas_ready();
void eden_webgpu_request_canvas(void* thread);
void eden_webgpu_canvas_size(u32* width, u32* height);
}

namespace WebGPU {

namespace {
constexpr char CANVAS_SELECTOR[] = "#edenCanvas";

void RunFunction(void* arg) {
    (*static_cast<const std::function<void()>*>(arg))();
}

std::string_view ToStringView(wgpu::StringView view) {
    return view.data ? std::string_view{view.data, view.length == WGPU_STRLEN ? std::strlen(view.data) : view.length}
                     : std::string_view{};
}
} // Anonymous namespace

Thread& Thread::Get() {
    static Thread instance;
    return instance;
}

bool Thread::Start() {
    if (init_state.load() == 0 && thread == pthread_t{}) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 4 * 1024 * 1024);
        const int result = pthread_create(&thread, &attr, ThreadEntry, this);
        pthread_attr_destroy(&attr);
        if (result != 0) {
            LOG_CRITICAL(Render, "Failed to create the WebGPU thread: {}", result);
            return false;
        }
    }
    u32 state;
    while ((state = init_state.load()) == 0) {
        emscripten_futex_wait(&init_state, 0, INFINITY);
    }
    return state == 1;
}

bool Thread::IsCurrent() const {
    return pthread_equal(pthread_self(), thread);
}

void Thread::RunSync(const std::function<void()>& func) {
    if (IsCurrent()) {
        func();
        return;
    }
    emscripten_proxy_sync(emscripten_proxy_get_system_queue(), thread, RunFunction,
                          const_cast<std::function<void()>*>(&func));
}

void Thread::SetPump(Pump new_pump) {
    RunSync([&] {
        pump = std::move(new_pump);
        if (pump && !pump_scheduled) {
            pump_scheduled = true;
            emscripten_set_immediate(OnImmediate, this);
        }
    });
}

void Thread::Notify() {
    work_counter.fetch_add(1, std::memory_order_release);
    emscripten_futex_wake(&work_counter, INT32_MAX);
}

void* Thread::ThreadEntry(void* arg) {
    auto* self = static_cast<Thread*>(arg);
    Common::SetCurrentThreadName("WebGPU");
    eden_webgpu_install_canvas_listener();
    eden_webgpu_request_canvas(reinterpret_cast<void*>(pthread_self()));
    self->InitDevice();
    // Everything else happens in event loop callbacks; keep the thread alive between them.
    emscripten_runtime_keepalive_push();
    emscripten_unwind_to_js_event_loop();
}

void Thread::InitDevice() {
    wgpu::InstanceDescriptor instance_desc{};
    instance = wgpu::CreateInstance(&instance_desc);
    if (!instance) {
        FinishInit(false);
        return;
    }
    wgpu::RequestAdapterOptions options{};
    options.powerPreference = wgpu::PowerPreference::HighPerformance;
    instance.RequestAdapter(
        &options, wgpu::CallbackMode::AllowSpontaneous,
        [this](wgpu::RequestAdapterStatus status, wgpu::Adapter new_adapter, wgpu::StringView message) {
            if (status != wgpu::RequestAdapterStatus::Success) {
                LOG_CRITICAL(Render, "WebGPU adapter request failed: {}", ToStringView(message));
                FinishInit(false);
                return;
            }
            adapter = std::move(new_adapter);

            std::vector<wgpu::FeatureName> features;
            for (const auto feature : {
                     wgpu::FeatureName::TextureCompressionBC,
                     wgpu::FeatureName::TextureCompressionASTC,
                     wgpu::FeatureName::TextureCompressionETC2,
                     wgpu::FeatureName::Depth32FloatStencil8,
                     wgpu::FeatureName::DepthClipControl,
                     wgpu::FeatureName::DualSourceBlending,
                     wgpu::FeatureName::Float32Filterable,
                     wgpu::FeatureName::Float32Blendable,
                     wgpu::FeatureName::ClipDistances,
                     wgpu::FeatureName::IndirectFirstInstance,
                     wgpu::FeatureName::RG11B10UfloatRenderable,
                     wgpu::FeatureName::TextureComponentSwizzle,
                     wgpu::FeatureName::TextureFormatsTier1,
                     wgpu::FeatureName::Unorm16TextureFormats,
                 }) {
                if (adapter.HasFeature(feature)) {
                    features.push_back(feature);
                }
            }
            wgpu::Limits limits{};
            adapter.GetLimits(&limits);

            wgpu::DeviceDescriptor desc{};
            desc.requiredFeatureCount = features.size();
            desc.requiredFeatures = features.data();
            desc.requiredLimits = &limits;
            desc.SetUncapturedErrorCallback(
                [](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView message) {
                    LOG_ERROR(Render, "WebGPU error {}: {}", static_cast<u32>(type),
                              ToStringView(message));
                });
            desc.SetDeviceLostCallback(
                wgpu::CallbackMode::AllowSpontaneous,
                [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView message) {
                    LOG_CRITICAL(Render, "WebGPU device lost ({}): {}", static_cast<u32>(reason),
                                 ToStringView(message));
                });
            adapter.RequestDevice(
                &desc, wgpu::CallbackMode::AllowSpontaneous,
                [this](wgpu::RequestDeviceStatus device_status, wgpu::Device new_device,
                       wgpu::StringView device_message) {
                    if (device_status != wgpu::RequestDeviceStatus::Success) {
                        LOG_CRITICAL(Render, "WebGPU device request failed: {}",
                                     ToStringView(device_message));
                        FinishInit(false);
                        return;
                    }
                    device = std::move(new_device);
                    queue = device.GetQueue();
                    InitSurface();
                });
        });
}

void Thread::InitSurface() {
    if (!eden_webgpu_canvas_ready()) {
        // The canvas is posted from the main thread; wait for the message to arrive.
        emscripten_async_call([](void* arg) { static_cast<Thread*>(arg)->InitSurface(); }, this, 5);
        return;
    }
    wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector selector{};
    selector.selector = CANVAS_SELECTOR;
    wgpu::SurfaceDescriptor surface_desc{};
    surface_desc.nextInChain = &selector;
    surface = instance.CreateSurface(&surface_desc);
    eden_webgpu_canvas_size(&surface_width, &surface_height);

    wgpu::SurfaceCapabilities capabilities{};
    surface.GetCapabilities(adapter, &capabilities);
    if (capabilities.formatCount > 0) {
        surface_format = capabilities.formats[0];
    }
    wgpu::SurfaceConfiguration config{};
    config.device = device;
    config.format = surface_format;
    config.usage = wgpu::TextureUsage::RenderAttachment;
    config.width = surface_width;
    config.height = surface_height;
    config.alphaMode = wgpu::CompositeAlphaMode::Opaque;
    config.presentMode = wgpu::PresentMode::Fifo;
    surface.Configure(&config);
    LOG_INFO(Render, "WebGPU ready: surface {}x{}", surface_width, surface_height);
    FinishInit(true);
}

void Thread::FinishInit(bool ok) {
    init_state.store(ok ? 1 : 2);
    emscripten_futex_wake(&init_state, INT32_MAX);
}

void Thread::OnImmediate(void* user_data) {
    static_cast<Thread*>(user_data)->RunPump();
}

void Thread::OnWake(int32_t*, uint32_t, int, void* user_data) {
    static_cast<Thread*>(user_data)->RunPump();
}

void Thread::RunPump() {
    pump_scheduled = false;
    if (!pump) {
        return;
    }
    const u32 seen = work_counter.load(std::memory_order_acquire);
    bool more = false;
    try {
        more = pump();
    } catch (const std::exception& e) {
        LOG_CRITICAL(Render, "WebGPU thread: unhandled exception: {}", e.what());
        more = true;
    }
    if (more) {
        pump_scheduled = true;
        emscripten_async_call(OnImmediate, this, 0);
        return;
    }
    pump_scheduled = true;
    const auto token = emscripten_atomic_wait_async(&work_counter, seen, OnWake, this,
                                                    EMSCRIPTEN_WAIT_ASYNC_INFINITY);
    if (!EMSCRIPTEN_IS_VALID_WAIT_TOKEN(token)) {
        // The counter moved while pumping; run again right away.
        emscripten_set_immediate(OnImmediate, this);
    }
}

} // namespace WebGPU
