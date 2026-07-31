#include "tatara/text/chat_template.h"

#include "tatara/model/sha256.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace tatara::text {
namespace {

constexpr std::uint64_t kMaximumTemplateBytes = 1ULL << 20U;
constexpr std::string_view kMessageStart = "<|im_start|>";
constexpr std::string_view kMessageEnd = "<|im_end|>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kToolResponseStart = "<tool_response>";
constexpr std::string_view kToolResponseEnd = "</tool_response>";

bool safe_relative_path(std::string_view value) {
    if (value.empty()) {
        return false;
    }
    const std::filesystem::path path(value);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) {
        return false;
    }
    for (const auto& component : path) {
        if (component == ".." || component == ".") {
            return false;
        }
    }
    return true;
}

ChatTemplateLoadResult load_error(ChatTemplateLoadError error,
                                  std::string detail) {
    return {
        .chat_template = nullptr,
        .error = error,
        .detail = std::move(detail),
    };
}

ChatRenderResult render_error(ChatTemplateError error, std::string detail) {
    return {
        .prompt = {},
        .error = error,
        .detail = std::move(detail),
    };
}

struct TrimResult {
    std::string value;
    bool valid_utf8;
};

TrimResult trim(std::string_view value) {
    @autoreleasepool {
        NSString* input = [[NSString alloc] initWithBytes:value.data()
                                                   length:value.size()
                                                 encoding:NSUTF8StringEncoding];
        if (input == nil) {
            return {{}, false};
        }
        NSString* trimmed = [input
            stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
        NSData* data =
            [trimmed dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
        if (data == nil) {
            return {{}, false};
        }
        return {
            std::string(static_cast<const char*>(data.bytes), data.length),
            true,
        };
    }
}

void strip_newlines(std::string& value, bool left, bool right) {
    if (right) {
        while (!value.empty() && value.back() == '\n') {
            value.pop_back();
        }
    }
    if (left) {
        const std::size_t beginning = value.find_first_not_of('\n');
        value.erase(0, beginning == std::string::npos ? value.size() : beginning);
    }
}

class BoundedPrompt {
  public:
    explicit BoundedPrompt(std::size_t limit) : limit_(limit) {
        value_.reserve(std::min<std::size_t>(limit, 4096));
    }

    bool append(std::string_view value) {
        if (value.size() > limit_ - value_.size()) {
            return false;
        }
        value_.append(value);
        return true;
    }

    std::string take() && { return std::move(value_); }

  private:
    std::size_t limit_;
    std::string value_;
};

bool is_tool_response(std::string_view value) {
    return value.starts_with(kToolResponseStart) &&
           value.ends_with(kToolResponseEnd);
}

} // namespace

ChatTemplateLoadResult Qwen36ChatTemplate::load(
    const model::qwen36::TokenizerSpec& specification,
    const std::filesystem::path& artifact_root) {
    try {
        if (specification.template_kind !=
            model::qwen36::ChatTemplateKind::Qwen36TextV1) {
            return load_error(ChatTemplateLoadError::UnsupportedTemplate,
                              "generated chat-template family is unsupported");
        }
        if (!safe_relative_path(specification.template_path)) {
            return load_error(ChatTemplateLoadError::UnsafePath,
                              "template path is not a safe relative path");
        }
        if (specification.template_size_bytes == 0) {
            return load_error(ChatTemplateLoadError::EmptyFile,
                              "template contract declares an empty file");
        }
        if (specification.template_size_bytes > kMaximumTemplateBytes) {
            return load_error(ChatTemplateLoadError::FileTooLarge,
                              "template contract exceeds 1 MiB");
        }
        const std::filesystem::path path =
            artifact_root / std::filesystem::path(specification.template_path);
        std::error_code filesystem_error;
        const auto status = std::filesystem::symlink_status(path, filesystem_error);
        if (filesystem_error || !std::filesystem::exists(status)) {
            return load_error(ChatTemplateLoadError::FileUnavailable,
                              "template file is unavailable");
        }
        if (std::filesystem::is_symlink(status) ||
            !std::filesystem::is_regular_file(status)) {
            return load_error(ChatTemplateLoadError::WrongFileType,
                              "template must be a regular non-symlink file");
        }
        const std::uintmax_t size =
            std::filesystem::file_size(path, filesystem_error);
        if (filesystem_error) {
            return load_error(ChatTemplateLoadError::FileUnavailable,
                              "template size is unavailable");
        }
        if (size != specification.template_size_bytes) {
            return load_error(ChatTemplateLoadError::FileSizeChanged,
                              "template size differs from the generated contract");
        }
        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        std::ifstream stream(path, std::ios::binary);
        if (!stream) {
            return load_error(ChatTemplateLoadError::FileUnavailable,
                              "template could not be opened");
        }
        stream.read(reinterpret_cast<char*>(bytes.data()),
                    static_cast<std::streamsize>(bytes.size()));
        if (!stream ||
            stream.gcount() != static_cast<std::streamsize>(bytes.size())) {
            return load_error(ChatTemplateLoadError::FileSizeChanged,
                              "template changed while it was read");
        }
        model::Sha256 hasher;
        hasher.update(bytes);
        if (model::sha256_hex(hasher.finish()) != specification.template_sha256) {
            return load_error(ChatTemplateLoadError::DigestMismatch,
                              "template digest differs from the generated contract");
        }
        return {
            .chat_template =
                std::unique_ptr<Qwen36ChatTemplate>(new Qwen36ChatTemplate()),
            .error = ChatTemplateLoadError::None,
            .detail = {},
        };
    } catch (const std::bad_alloc&) {
        return load_error(ChatTemplateLoadError::AllocationFailure,
                          "chat-template allocation failed");
    }
}

ChatRenderResult Qwen36ChatTemplate::render(
    const std::vector<ChatMessage>& messages, bool enable_thinking,
    std::size_t maximum_output_bytes) const {
    try {
        if (messages.empty()) {
            return render_error(ChatTemplateError::EmptyMessages,
                                "messages must be non-empty");
        }
        std::vector<std::string> content;
        content.reserve(messages.size());
        for (const ChatMessage& message : messages) {
            TrimResult trimmed = trim(message.content);
            if (!trimmed.valid_utf8) {
                return render_error(ChatTemplateError::InvalidUtf8,
                                    "message content is not valid UTF-8");
            }
            content.push_back(std::move(trimmed.value));
        }

        std::size_t last_query = messages.size();
        for (std::size_t reverse = messages.size(); reverse > 0; --reverse) {
            const std::size_t index = reverse - 1;
            if (messages[index].role == "user" &&
                !is_tool_response(content[index])) {
                last_query = index;
                break;
            }
        }
        if (last_query == messages.size()) {
            return render_error(ChatTemplateError::MissingUserQuery,
                                "no user query found in messages");
        }

        BoundedPrompt prompt(maximum_output_bytes);
        for (std::size_t index = 0; index < messages.size(); ++index) {
            const ChatMessage& message = messages[index];
            std::string rendered_content = content[index];
            if (message.role == "system") {
                if (index != 0) {
                    return render_error(
                        ChatTemplateError::MisplacedSystemMessage,
                        "system message must be at the beginning");
                }
                if (!prompt.append(kMessageStart) || !prompt.append("system\n") ||
                    !prompt.append(rendered_content) ||
                    !prompt.append(kMessageEnd) || !prompt.append("\n")) {
                    return render_error(ChatTemplateError::OutputLimitExceeded,
                                        "rendered prompt exceeds the byte limit");
                }
                continue;
            }
            if (message.role == "user") {
                if (!prompt.append(kMessageStart) || !prompt.append("user\n") ||
                    !prompt.append(rendered_content) ||
                    !prompt.append(kMessageEnd) || !prompt.append("\n")) {
                    return render_error(ChatTemplateError::OutputLimitExceeded,
                                        "rendered prompt exceeds the byte limit");
                }
                continue;
            }
            if (message.role != "assistant") {
                return render_error(
                    ChatTemplateError::UnsupportedRole,
                    "text subset supports system, user, and assistant roles");
            }

            std::string reasoning;
            if (message.reasoning_content.has_value()) {
                TrimResult trimmed = trim(*message.reasoning_content);
                if (!trimmed.valid_utf8) {
                    return render_error(ChatTemplateError::InvalidUtf8,
                                        "assistant reasoning is not valid UTF-8");
                }
                reasoning = std::move(trimmed.value);
            } else {
                const std::size_t first_close =
                    rendered_content.find(kThinkEnd);
                if (first_close != std::string::npos) {
                    reasoning = rendered_content.substr(0, first_close);
                    strip_newlines(reasoning, false, true);
                    const std::size_t last_open = reasoning.rfind(kThinkStart);
                    if (last_open != std::string::npos) {
                        reasoning.erase(0, last_open + kThinkStart.size());
                    }
                    strip_newlines(reasoning, true, false);
                    const std::size_t last_close =
                        rendered_content.rfind(kThinkEnd);
                    rendered_content.erase(
                        0, last_close + kThinkEnd.size());
                    strip_newlines(rendered_content, true, false);
                    TrimResult trimmed = trim(reasoning);
                    if (!trimmed.valid_utf8) {
                        return render_error(ChatTemplateError::InvalidUtf8,
                                            "assistant reasoning is not valid UTF-8");
                    }
                    reasoning = std::move(trimmed.value);
                }
            }
            if (!prompt.append(kMessageStart) || !prompt.append("assistant\n")) {
                return render_error(ChatTemplateError::OutputLimitExceeded,
                                    "rendered prompt exceeds the byte limit");
            }
            if (index > last_query &&
                (!prompt.append("<think>\n") || !prompt.append(reasoning) ||
                 !prompt.append("\n</think>\n\n"))) {
                return render_error(ChatTemplateError::OutputLimitExceeded,
                                    "rendered prompt exceeds the byte limit");
            }
            if (!prompt.append(rendered_content) || !prompt.append(kMessageEnd) ||
                !prompt.append("\n")) {
                return render_error(ChatTemplateError::OutputLimitExceeded,
                                    "rendered prompt exceeds the byte limit");
            }
        }

        if (!prompt.append("<|im_start|>assistant\n") ||
            !(enable_thinking ? prompt.append("<think>\n")
                              : prompt.append("<think>\n\n</think>\n\n"))) {
            return render_error(ChatTemplateError::OutputLimitExceeded,
                                "rendered prompt exceeds the byte limit");
        }
        return {
            .prompt = std::move(prompt).take(),
            .error = ChatTemplateError::None,
            .detail = {},
        };
    } catch (const std::bad_alloc&) {
        return render_error(ChatTemplateError::AllocationFailure,
                            "chat-template request allocation failed");
    }
}

Qwen36ChatTemplate::Qwen36ChatTemplate(Qwen36ChatTemplate&&) noexcept = default;
Qwen36ChatTemplate& Qwen36ChatTemplate::operator=(
    Qwen36ChatTemplate&&) noexcept = default;
Qwen36ChatTemplate::~Qwen36ChatTemplate() = default;

} // namespace tatara::text
