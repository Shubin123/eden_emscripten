// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <string>

#include "common/common_types.h"

namespace WebGPU {

/// Pipeline cache counters, readable from any thread.
struct PipelineStats {
    /// Graphics pipeline lookups by draws, and those that created a new pipeline.
    std::atomic<u64> lookups{};
    std::atomic<u64> misses{};
    /// Pipeline variants built: guest shader translation to WGSL plus the creation request.
    std::atomic<u64> builds{};
    std::atomic<u64> build_us{};
    std::atomic<u64> build_failures{};
    /// Asynchronous pipeline creations completed by the browser, and the wall time from request
    /// to completion.
    std::atomic<u64> creations{};
    std::atomic<u64> creation_us{};
    std::atomic<u64> creation_failures{};

    void Reset() {
        for (auto* counter : {&lookups, &misses, &builds, &build_us, &build_failures, &creations,
                              &creation_us, &creation_failures}) {
            counter->store(0, std::memory_order_relaxed);
        }
    }
};

inline PipelineStats pipeline_stats;

} // namespace WebGPU
