// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dynarmic/common/spin_lock.h"

namespace Dynarmic {

void SpinLock::Lock() noexcept {
    while (__atomic_exchange_n(&storage, 1, __ATOMIC_ACQUIRE) != 0) {
        while (__atomic_load_n(&storage, __ATOMIC_RELAXED) != 0) {
        }
    }
}

void SpinLock::Unlock() noexcept {
    __atomic_store_n(&storage, 0, __ATOMIC_RELEASE);
}

}  // namespace Dynarmic
