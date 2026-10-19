// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Hands the page canvas to the WebGPU thread as an OffscreenCanvas.
addToLibrary({
  eden_webgpu_install_canvas_listener__deps: ['$specialHTMLTargets'],
  eden_webgpu_install_canvas_listener__sig: 'v',
  eden_webgpu_install_canvas_listener: () => {
    // Intercept before Emscripten's handler, which rejects unknown commands.
    const handler = self.onmessage;
    self.onmessage = (e) => {
      if (e.data && e.data.cmd === 'edenCanvas') {
        specialHTMLTargets['#edenCanvas'] = e.data.canvas;
        return;
      }
      handler(e);
    };
  },
  eden_webgpu_canvas_ready__deps: ['$specialHTMLTargets'],
  eden_webgpu_canvas_ready__sig: 'i',
  eden_webgpu_canvas_ready: () => specialHTMLTargets['#edenCanvas'] ? 1 : 0,
  eden_webgpu_canvas_size__deps: ['$specialHTMLTargets'],
  eden_webgpu_canvas_size__sig: 'vpp',
  eden_webgpu_canvas_size: (width, height) => {
    const canvas = specialHTMLTargets['#edenCanvas'];
    {{{ makeSetValue('width', 0, 'canvas.width', 'i32') }}};
    {{{ makeSetValue('height', 0, 'canvas.height', 'i32') }}};
  },
  eden_webgpu_request_canvas__proxy: 'sync',
  eden_webgpu_request_canvas__sig: 'vp',
  eden_webgpu_request_canvas: (thread) => {
    const canvas = (typeof Module !== 'undefined' && Module['canvas']) || document.getElementById('canvas');
    const offscreen = canvas.transferControlToOffscreen();
    PThread.pthreads[thread].postMessage({ cmd: 'edenCanvas', canvas: offscreen }, [offscreen]);
  },
});
