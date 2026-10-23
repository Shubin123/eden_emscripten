// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>

#include "input_common/drivers/virtual_gamepad.h"

namespace InputCommon {
constexpr std::size_t PlayerIndexCount = 10;

VirtualGamepad::VirtualGamepad(std::string input_engine_) : InputEngine(std::move(input_engine_)) {
    for (std::size_t i = 0; i < PlayerIndexCount; i++) {
        PreSetController(GetIdentifier(i));
    }
}

void VirtualGamepad::SetButtonState(std::size_t player_index, int button_id, bool value) {
    if (player_index > PlayerIndexCount) {
        return;
    }
    const auto identifier = GetIdentifier(player_index);
    SetButton(identifier, button_id, value);
}

void VirtualGamepad::SetButtonState(std::size_t player_index, VirtualButton button_id, bool value) {
    SetButtonState(player_index, static_cast<int>(button_id), value);
}

void VirtualGamepad::SetStickPosition(std::size_t player_index, int axis_id, float x_value,
                                      float y_value) {
    if (player_index > PlayerIndexCount) {
        return;
    }
    const auto identifier = GetIdentifier(player_index);
    SetAxis(identifier, axis_id * 2, x_value);
    SetAxis(identifier, (axis_id * 2) + 1, y_value);
}

void VirtualGamepad::SetStickPosition(std::size_t player_index, VirtualStick axis_id, float x_value,
                                      float y_value) {
    SetStickPosition(player_index, static_cast<int>(axis_id), x_value, y_value);
}

void VirtualGamepad::SetMotionState(std::size_t player_index, u64 delta_timestamp, float gyro_x,
                                    float gyro_y, float gyro_z, float accel_x, float accel_y,
                                    float accel_z) {
    const auto identifier = GetIdentifier(player_index);
    const BasicMotion motion_data{
        .gyro_x = gyro_x,
        .gyro_y = gyro_y,
        .gyro_z = gyro_z,
        .accel_x = accel_x,
        .accel_y = accel_y,
        .accel_z = accel_z,
        .delta_timestamp = delta_timestamp,
    };
    SetMotion(identifier, 0, motion_data);
}

void VirtualGamepad::ResetControllers() {
    for (std::size_t i = 0; i < PlayerIndexCount; i++) {
        SetStickPosition(i, VirtualStick::Left, 0.0f, 0.0f);
        SetStickPosition(i, VirtualStick::Right, 0.0f, 0.0f);

        SetButtonState(i, VirtualButton::ButtonA, false);
        SetButtonState(i, VirtualButton::ButtonB, false);
        SetButtonState(i, VirtualButton::ButtonX, false);
        SetButtonState(i, VirtualButton::ButtonY, false);
        SetButtonState(i, VirtualButton::StickL, false);
        SetButtonState(i, VirtualButton::StickR, false);
        SetButtonState(i, VirtualButton::TriggerL, false);
        SetButtonState(i, VirtualButton::TriggerR, false);
        SetButtonState(i, VirtualButton::TriggerZL, false);
        SetButtonState(i, VirtualButton::TriggerZR, false);
        SetButtonState(i, VirtualButton::ButtonPlus, false);
        SetButtonState(i, VirtualButton::ButtonMinus, false);
        SetButtonState(i, VirtualButton::ButtonLeft, false);
        SetButtonState(i, VirtualButton::ButtonUp, false);
        SetButtonState(i, VirtualButton::ButtonRight, false);
        SetButtonState(i, VirtualButton::ButtonDown, false);
        SetButtonState(i, VirtualButton::ButtonSL, false);
        SetButtonState(i, VirtualButton::ButtonSR, false);
        SetButtonState(i, VirtualButton::ButtonHome, false);
        SetButtonState(i, VirtualButton::ButtonCapture, false);
    }
}

PadIdentifier VirtualGamepad::GetIdentifier(std::size_t player_index) const {
    return {
        .guid = Common::UUID{},
        .port = player_index,
        .pad = 0,
    };
}

namespace {
u64 PackRumble(const Common::Input::VibrationStatus& vibration) {
    const auto unit = [](float value) { return u64(std::clamp(value, 0.0f, 1.0f) * 65535.0f); };
    return unit(vibration.low_amplitude) | (unit(vibration.high_amplitude) << 16);
}
} // Anonymous namespace

Common::Input::DriverResult VirtualGamepad::SetVibration(
    const PadIdentifier& identifier, const Common::Input::VibrationStatus& vibration) {
    if (identifier.port >= rumble.size()) {
        return Common::Input::DriverResult::InvalidParameters;
    }
    // The left half of a controller vibrates first; the right half overwrites the other slot.
    auto& sides = rumble[identifier.port];
    const std::size_t side = identifier.pad == 1 ? 1 : 0;
    sides[side].store(PackRumble(vibration), std::memory_order_relaxed);
    return Common::Input::DriverResult::Success;
}

bool VirtualGamepad::IsVibrationEnabled(const PadIdentifier&) {
    return true;
}

VirtualGamepad::Rumble VirtualGamepad::GetRumble(std::size_t player_index) const {
    if (player_index >= rumble.size()) {
        return {};
    }
    Rumble result{};
    for (const auto& side : rumble[player_index]) {
        const u64 packed = side.load(std::memory_order_relaxed);
        result.low_amplitude = std::max(result.low_amplitude, float(packed & 0xFFFF) / 65535.0f);
        result.high_amplitude =
            std::max(result.high_amplitude, float((packed >> 16) & 0xFFFF) / 65535.0f);
    }
    return result;
}

} // namespace InputCommon
