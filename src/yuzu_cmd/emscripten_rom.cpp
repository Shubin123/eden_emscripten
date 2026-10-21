// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

#include <emscripten.h>
#include <emscripten/threading.h>

#include "common/logging.h"
#include "core/file_sys/vfs/vfs_real.h"
#include "yuzu_cmd/emscripten_rom.h"

namespace {
constexpr size_t IoBufferSize = 4 * 1024 * 1024;
struct alignas(8) IoControl {
    std::atomic<int32_t> status{0};
    int32_t length = 0;
    int32_t result = 0;
    int32_t reserved = 0;
    uint64_t offset = 0;
};
static_assert(offsetof(IoControl, offset) == 16);
IoControl control;
std::array<u8, IoBufferSize> io_buffer;
std::mutex io_mutex;
std::string rom_path;
size_t rom_size = 0;
bool io_failed = false;

class BrowserRomFile final : public FileSys::VfsFile {
public:
    BrowserRomFile(std::string path, size_t size_)
        : name{std::filesystem::path{path}.filename().string()}, size{size_} {}
    std::string GetName() const override {
        return name;
    }
    size_t GetSize() const override {
        return size;
    }
    bool Resize(size_t) override {
        return false;
    }
    FileSys::VirtualDir GetContainingDirectory() const override {
        return nullptr;
    }
    bool IsWritable() const override {
        return false;
    }
    bool IsReadable() const override {
        return true;
    }
    size_t Write(const u8*, size_t, size_t) override {
        return 0;
    }
    bool Rename(std::string_view) override {
        return false;
    }

    size_t Read(u8* data, size_t length, size_t offset) const override {
        if (offset >= size || length == 0 || emscripten_is_main_browser_thread())
            return 0;
        std::scoped_lock lock{io_mutex};
        if (io_failed)
            return 0;
        const size_t requested = std::min(length, size - offset);
        size_t copied = 0;
        while (copied < requested) {
            control.offset = offset + copied;
            control.length = static_cast<int32_t>(std::min(requested - copied, IoBufferSize));
            control.result = 0;
            control.status.store(1);
            emscripten_futex_wake(&control.status, 1);
            const double deadline = emscripten_get_now() + 10'000.0;
            while (control.status.load() == 1) {
                const double remaining = deadline - emscripten_get_now();
                if (remaining <= 0)
                    break;
                emscripten_futex_wait(&control.status, 1, remaining);
            }
            if (control.status.load() != 2 || control.result < 0 ||
                control.result > control.length) {
                io_failed = true;
                LOG_ERROR(Frontend, "Browser ROM read failed at {} ({} bytes)", control.offset,
                          control.length);
                return copied;
            }
            const size_t count = static_cast<size_t>(control.result);
            std::memcpy(data + copied, io_buffer.data(), count);
            copied += count;
            control.status.store(0);
            if (count < static_cast<size_t>(control.length))
                break;
        }
        return copied;
    }

private:
    std::string name;
    size_t size;
};

class BrowserRomFilesystem final : public FileSys::RealVfsFilesystem {
public:
    BrowserRomFilesystem(std::string path_, size_t size)
        : path{std::move(path_)}, file{std::make_shared<BrowserRomFile>(path, size)} {}
    FileSys::VfsEntryType GetEntryType(std::string_view candidate) const override {
        return candidate == path ? FileSys::VfsEntryType::File
                                 : RealVfsFilesystem::GetEntryType(candidate);
    }
    FileSys::VirtualFile OpenFile(std::string_view candidate, FileSys::OpenMode perms) override {
        if (candidate == path)
            return perms == FileSys::OpenMode::Read ? file : nullptr;
        return RealVfsFilesystem::OpenFile(candidate, perms);
    }

private:
    std::string path;
    FileSys::VirtualFile file;
};
} // namespace

std::shared_ptr<FileSys::VfsFilesystem> CreateBrowserRomFilesystem(std::string_view path) {
    std::scoped_lock lock{io_mutex};
    if (path != rom_path || rom_size == 0)
        return nullptr;
    return std::make_shared<BrowserRomFilesystem>(rom_path, rom_size);
}

extern "C" {
EMSCRIPTEN_KEEPALIVE void* eden_get_rom_io_control() {
    return &control;
}
EMSCRIPTEN_KEEPALIVE void* eden_get_rom_io_buffer() {
    return io_buffer.data();
}
EMSCRIPTEN_KEEPALIVE int eden_get_rom_io_buffer_size() {
    return static_cast<int>(IoBufferSize);
}
EMSCRIPTEN_KEEPALIVE int eden_register_browser_rom(const char* path, double size) {
    if (!path || !path[0] || !std::isfinite(size) || size <= 0 || std::floor(size) != size ||
        size > 9007199254740991.0)
        return -1;
    std::scoped_lock lock{io_mutex};
    rom_path = path;
    rom_size = static_cast<size_t>(size);
    io_failed = false;
    control.status = 0;
    return 0;
}
} // extern "C"
