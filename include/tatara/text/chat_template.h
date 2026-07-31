#pragma once

#include "tatara/model/qwen36_plan.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tatara::text {

enum class ChatTemplateLoadError : std::uint8_t {
    None,
    UnsupportedTemplate,
    UnsafePath,
    FileUnavailable,
    WrongFileType,
    EmptyFile,
    FileTooLarge,
    FileSizeChanged,
    DigestMismatch,
    AllocationFailure,
};

enum class ChatTemplateError : std::uint8_t {
    None,
    EmptyMessages,
    InvalidUtf8,
    MissingUserQuery,
    MisplacedSystemMessage,
    UnsupportedRole,
    OutputLimitExceeded,
    ArithmeticOverflow,
    AllocationFailure,
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::optional<std::string> reasoning_content;
};

struct ChatRenderResult {
    std::string prompt;
    ChatTemplateError error = ChatTemplateError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == ChatTemplateError::None;
    }
};

class Qwen36ChatTemplate;
struct ChatTemplateLoadResult;

class Qwen36ChatTemplate final {
  public:
    Qwen36ChatTemplate(const Qwen36ChatTemplate&) = delete;
    Qwen36ChatTemplate& operator=(const Qwen36ChatTemplate&) = delete;
    Qwen36ChatTemplate(Qwen36ChatTemplate&&) noexcept;
    Qwen36ChatTemplate& operator=(Qwen36ChatTemplate&&) noexcept;
    ~Qwen36ChatTemplate();

    static ChatTemplateLoadResult load(
        const model::qwen36::TokenizerSpec& specification,
        const std::filesystem::path& artifact_root);

    ChatRenderResult render(const std::vector<ChatMessage>& messages,
                            bool enable_thinking,
                            std::size_t maximum_output_bytes) const;

  private:
    Qwen36ChatTemplate() = default;
};

struct ChatTemplateLoadResult {
    std::unique_ptr<Qwen36ChatTemplate> chat_template;
    ChatTemplateLoadError error = ChatTemplateLoadError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == ChatTemplateLoadError::None && chat_template != nullptr;
    }
};

} // namespace tatara::text
