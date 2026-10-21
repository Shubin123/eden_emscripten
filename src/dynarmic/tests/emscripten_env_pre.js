// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Forward EDEN_* environment variables (JIT switches) from Node.js to the program environment.
Module['preRun'] = Module['preRun'] || [];
Module['preRun'].push(() => {
    for (const [key, value] of Object.entries(process.env)) {
        if (key.startsWith('EDEN_')) {
            ENV[key] = value;
        }
    }
});
