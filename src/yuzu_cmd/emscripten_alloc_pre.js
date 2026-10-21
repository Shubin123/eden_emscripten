// SPDX-License-Identifier: GPL-3.0-or-later

const edenOriginalMemoryGrow = WebAssembly.Memory.prototype.grow;
WebAssembly.Memory.prototype.grow = function(pages) {
    console.debug('Eden memory growth begin: pages=' + pages + ' bytes=' + this.buffer.byteLength);
    try {
        const previous = edenOriginalMemoryGrow.call(this, pages);
        console.debug('Eden memory growth complete: bytes=' + this.buffer.byteLength);
        return previous;
    } catch (error) {
        console.error('Eden memory growth failed: pages=' + pages + ' bytes=' + this.buffer.byteLength + ' error=' + error);
        throw error;
    }
};

if (typeof GPUDevice !== 'undefined') {
    const edenOriginalCreateTexture = GPUDevice.prototype.createTexture;
    GPUDevice.prototype.createTexture = function(descriptor) {
        try {
            return edenOriginalCreateTexture.call(this, descriptor);
        } catch (error) {
            console.error('Eden texture allocation failed: ' + error + ' descriptor=' + JSON.stringify(descriptor));
            throw error;
        }
    };
}
