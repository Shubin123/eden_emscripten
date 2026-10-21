// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <memory>
#include <string_view>

namespace FileSys {
class VfsFilesystem;
}

std::shared_ptr<FileSys::VfsFilesystem> CreateBrowserRomFilesystem(std::string_view path);
