// SPDX-License-Identifier: GPL-3.0-or-later

const edenOriginalMemoryGrow = WebAssembly.Memory.prototype.grow;
WebAssembly.Memory.prototype.grow = function(pages) {
    try {
        return edenOriginalMemoryGrow.call(this, pages);
    } catch (error) {
        console.error('Eden memory growth failed: pages=' + pages + ' bytes=' + this.buffer.byteLength + ' error=' + error);
        throw error;
    }
};
