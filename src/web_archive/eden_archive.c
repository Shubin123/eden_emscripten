// SPDX-License-Identifier: GPL-3.0-or-later
// Streaming libarchive reader for the browser frontend. The archive is pulled through a
// synchronous JS callback (FileReaderSync in a worker), so neither the archive nor the
// extracted entry has to fit in memory.

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <archive.h>
#include <archive_entry.h>
#include <emscripten.h>

// Copies up to len bytes at offset into buf and returns the count, or -1 on failure.
EM_JS(int, js_read_at, (double offset, uint8_t* buf, int len), {
    return Module.readAt(offset, buf, len);
});

typedef struct {
    struct archive* archive;
    struct archive_entry* entry;
    int64_t size;
    int64_t pos;
    uint8_t buffer[1 << 20];
} Reader;

static la_ssize_t ReadCallback(struct archive* a, void* data, const void** out) {
    Reader* r = data;
    const int64_t left = r->size - r->pos;
    const int want = left < (int64_t)sizeof(r->buffer) ? (int)left : (int)sizeof(r->buffer);
    *out = r->buffer;
    if (want <= 0) {
        return 0;
    }
    const int n = js_read_at((double)r->pos, r->buffer, want);
    if (n < 0) {
        archive_set_error(a, EIO, "Failed to read the archive file");
        return -1;
    }
    r->pos += n;
    return n;
}

static la_int64_t SeekCallback(struct archive* a, void* data, la_int64_t offset, int whence) {
    Reader* r = data;
    const int64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? r->pos : r->size;
    const int64_t target = base + offset;
    if (target < 0 || target > r->size) {
        return ARCHIVE_FATAL;
    }
    r->pos = target;
    return target;
}

static la_int64_t SkipCallback(struct archive* a, void* data, la_int64_t request) {
    Reader* r = data;
    const int64_t left = r->size - r->pos;
    if (request > left) {
        request = left;
    }
    r->pos += request;
    return request;
}

EMSCRIPTEN_KEEPALIVE Reader* ea_open(double size) {
    Reader* r = calloc(1, sizeof(Reader));
    if (!r) {
        return NULL;
    }
    r->size = (int64_t)size;
    r->archive = archive_read_new();
    archive_read_support_filter_all(r->archive);
    archive_read_support_format_all(r->archive);
    // Lets single-file streams such as game.nsp.xz open as one entry. It only wins when no
    // real archive format matches.
    archive_read_support_format_raw(r->archive);
    archive_read_set_callback_data(r->archive, r);
    archive_read_set_read_callback(r->archive, ReadCallback);
    archive_read_set_seek_callback(r->archive, SeekCallback);
    archive_read_set_skip_callback(r->archive, SkipCallback);
    archive_read_open1(r->archive);
    return r;
}

// Advances to the next entry: 1 on success, 0 at the end, -1 on error.
EMSCRIPTEN_KEEPALIVE int ea_next(Reader* r) {
    const int status = archive_read_next_header(r->archive, &r->entry);
    if (status == ARCHIVE_EOF) {
        return 0;
    }
    return status >= ARCHIVE_WARN ? 1 : -1;
}

EMSCRIPTEN_KEEPALIVE const char* ea_entry_name(Reader* r) {
    const char* name = archive_entry_pathname_utf8(r->entry);
    return name ? name : archive_entry_pathname(r->entry);
}

// Returns -1 when the archive does not record the size, as with raw compressed streams.
EMSCRIPTEN_KEEPALIVE double ea_entry_size(Reader* r) {
    return archive_entry_size_is_set(r->entry) ? (double)archive_entry_size(r->entry) : -1.0;
}

EMSCRIPTEN_KEEPALIVE int ea_entry_is_file(Reader* r) {
    return archive_entry_filetype(r->entry) == AE_IFREG;
}

// Reads entry data: the byte count, 0 at the end of the entry, or -1 on error.
EMSCRIPTEN_KEEPALIVE int ea_read(Reader* r, void* buffer, int length) {
    const la_ssize_t n = archive_read_data(r->archive, buffer, length);
    return n < 0 ? -1 : (int)n;
}

EMSCRIPTEN_KEEPALIVE const char* ea_format(Reader* r) {
    const char* format = archive_format_name(r->archive);
    return format ? format : "unknown";
}

EMSCRIPTEN_KEEPALIVE const char* ea_error(Reader* r) {
    const char* error = archive_error_string(r->archive);
    return error ? error : "unknown archive error";
}

EMSCRIPTEN_KEEPALIVE void ea_close(Reader* r) {
    archive_read_free(r->archive);
    free(r);
}
