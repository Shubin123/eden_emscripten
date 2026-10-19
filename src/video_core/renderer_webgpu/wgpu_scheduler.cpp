// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_webgpu/wgpu_scheduler.h"
#include "video_core/renderer_webgpu/wgpu_thread.h"

namespace WebGPU {

void Scheduler::EnsureEncoder() {
    if (!encoder) {
        encoder = Thread::Get().device.CreateCommandEncoder();
    }
}

wgpu::CommandEncoder& Scheduler::Encoder() {
    EndRenderPass();
    EnsureEncoder();
    has_commands = true;
    return encoder;
}

wgpu::RenderPassEncoder& Scheduler::RenderPass(const RenderTargets& targets) {
    if (!pass || !(current_targets == targets)) {
        EndRenderPass();
        BeginPass(targets, nullptr);
    }
    return pass;
}

void Scheduler::Clear(const RenderTargets& targets, const ClearRequest& request) {
    EndRenderPass();
    BeginPass(targets, &request);
}

void Scheduler::BeginPass(const RenderTargets& targets, const ClearRequest* clear) {
    EnsureEncoder();
    std::array<wgpu::RenderPassColorAttachment, 8> colors{};
    size_t num_colors = 0;
    for (size_t i = 0; i < targets.colors.size(); ++i) {
        if (!targets.colors[i]) {
            continue;
        }
        num_colors = i + 1;
        auto& attachment = colors[i];
        attachment.view = targets.colors[i];
        attachment.storeOp = wgpu::StoreOp::Store;
        if (clear && (clear->color_mask & (1u << i))) {
            attachment.loadOp = wgpu::LoadOp::Clear;
            attachment.clearValue = clear->colors[i];
        } else {
            attachment.loadOp = wgpu::LoadOp::Load;
        }
    }
    wgpu::RenderPassDepthStencilAttachment depth{};
    wgpu::RenderPassDescriptor desc{};
    desc.colorAttachmentCount = num_colors;
    desc.colorAttachments = colors.data();
    if (targets.depth_stencil) {
        depth.view = targets.depth_stencil;
        depth.depthLoadOp = clear && clear->depth ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
        depth.depthStoreOp = wgpu::StoreOp::Store;
        depth.depthClearValue = clear ? clear->depth_value : 0.0f;
        if (targets.has_stencil) {
            depth.stencilLoadOp = clear && clear->stencil ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
            depth.stencilStoreOp = wgpu::StoreOp::Store;
            depth.stencilClearValue = clear ? clear->stencil_value : 0;
        }
        desc.depthStencilAttachment = &depth;
    }
    pass = encoder.BeginRenderPass(&desc);
    current_targets = targets;
    has_commands = true;
    ++pass_serial;
}

void Scheduler::EndRenderPass() {
    if (pass) {
        pass.End();
        pass = nullptr;
    }
}

void Scheduler::Flush() {
    EndRenderPass();
    if (encoder && has_commands) {
        const wgpu::CommandBuffer commands = encoder.Finish();
        Thread::Get().queue.Submit(1, &commands);
        ++submit_count;
    }
    encoder = nullptr;
    has_commands = false;
}

} // namespace WebGPU
