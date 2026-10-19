// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <functional>
#include <memory>

namespace boost::context::detail {
struct transfer_t;
}

namespace Common {

/**
 * Fiber class
 * a fiber is a userspace thread with it's own context. They can be used to
 * implement coroutines, emulated threading systems and certain asynchronous
 * patterns.
 *
 * This class implements fibers at a low level, thus allowing greater freedom
 * to implement such patterns. This fiber class is 'threadsafe' only one fiber
 * can be running at a time and threads will be locked while trying to yield to
 * a running fiber until it yields. WARNING exchanging two running fibers between
 * threads will cause a deadlock. In order to prevent a deadlock, each thread should
 * have an intermediary fiber, you switch to the intermediary fiber of the current
 * thread and then from it switch to the expected fiber. This way you can exchange
 * 2 fibers within 2 different threads.
 */
class Fiber {
public:
    Fiber(std::function<void()>&& entry_point_func);
    ~Fiber();

    Fiber(const Fiber&) = delete;
    Fiber& operator=(const Fiber&) = delete;

    Fiber(Fiber&&) = default;
    Fiber& operator=(Fiber&&) = default;

    /// Yields control from Fiber 'from' to Fiber 'to'
    /// Fiber 'from' must be the currently running fiber.
    static void YieldTo(std::weak_ptr<Fiber> weak_from, Fiber& to);
    [[nodiscard]] static std::shared_ptr<Fiber> ThreadToFiber();
    void SetRewindPoint(std::function<void()>&& rewind_func);
    /// Only call from main thread's fiber
    void Exit();

    /// Pointer-sized slot that follows the logical thread of execution across YieldTo. Where fibers
    /// are backed by host threads (Emscripten), thread-local state that must be shared by every
    /// fiber running on behalf of one host thread should be reached through this slot.
    [[nodiscard]] static void*& LogicalThreadContext();

    struct FiberImpl;
private:
    Fiber();
#ifdef __EMSCRIPTEN__
    std::shared_ptr<FiberImpl> impl;
#else
    void Start(boost::context::detail::transfer_t& transfer);
    std::unique_ptr<FiberImpl> impl;
#endif
};

} // namespace Common
