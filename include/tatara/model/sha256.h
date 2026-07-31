#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace tatara::model {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
  public:
    Sha256() noexcept;

    void update(std::span<const std::byte> bytes) noexcept;

    // Finalizes the current message and resets to a fresh hasher.
    Sha256Digest finish() noexcept;

  private:
    // Opaque storage for the system digest context; checked in the adapter.
    alignas(8) std::array<std::byte, 128> context_;
};

std::string sha256_hex(const Sha256Digest& digest);

} // namespace tatara::model
