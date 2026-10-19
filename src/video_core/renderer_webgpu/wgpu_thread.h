// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// WebGPU objects belong to the JS realm that created them, so every WebGPU call runs on one
// pthread that returns to its event loop between batches of work. This lets the canvas present
// and asynchronous WebGPU callbacks run.

#pragma once

#include <atomic>
#include <functional>

#include <pthread.h>
#include <webgpu/webgpu_cpp.h>

#include "common/common_types.h"

namespace WebGPU {

class Thread {
public:
    /// Work pumped on the thread. Returns true if it stopped with work left (time budget, a
    /// presented frame) and wants to run again after yielding to the event loop.
    using Pump = std::function<bool()>;

    static Thread& Get();

    /// Starts the thread and blocks until the device and the canvas surface are ready.
    /// Returns false if WebGPU is unavailable.
    bool Start();

    bool IsCurrent() const;

    /// Runs `func` on the WebGPU thread and waits for it.
    void RunSync(const std::function<void()>& func);

    /// Installs (or clears, with nullptr) the pump and runs it whenever Notify is called.
    void SetPump(Pump pump);

    /// Wakes the pump. Callable from any thread.
    void Notify();

    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::TextureFormat surface_format = wgpu::TextureFormat::BGRA8Unorm;
    u32 surface_width = 1280;
    u32 surface_height = 720;

private:
    static void* ThreadEntry(void* arg);
    static void OnWake(int32_t* address, uint32_t value, int result, void* user_data);
    static void OnImmediate(void* user_data);

    void InitDevice();
    void InitSurface();
    void FinishInit(bool ok);
    void RunPump();
    void WaitForWork();

    pthread_t thread{};
    std::atomic<u32> init_state{0}; ///< 0 pending, 1 ready, 2 failed
    std::atomic<u32> work_counter{0};
    Pump pump;
    bool pump_scheduled = false;
};

} // namespace WebGPU
