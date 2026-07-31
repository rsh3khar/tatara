#pragma once

#include "tatara/model/qwen36_plan.h"
#include "tatara/service/router.h"
#include "tatara/text/chat_template.h"
#include "tatara/text/tokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tatara::service {

enum class CompletionRequestError : std::uint8_t {
    None,
    BodyLimitExceeded,
    InvalidJson,
    MissingField,
    UnknownField,
    WrongFieldType,
    WrongModel,
    UnsupportedFeature,
    EmptyPrompt,
    InvalidTokenId,
    InvalidMaximumTokens,
    ContextExceeded,
    TokenizerFailure,
    TemplateFailure,
    AllocationFailure,
};

enum class PromptKind : std::uint8_t {
    Text,
    TokenIds,
    Chat,
};

struct CompletionRequestLimits {
    std::size_t maximum_body_bytes;
    std::size_t maximum_input_bytes;
    std::uint32_t maximum_prompt_tokens;
    std::uint32_t maximum_context_tokens;
    std::uint32_t default_maximum_tokens;
};

struct ParsedCompletionRequest {
    Route route = Route::Completions;
    PromptKind prompt_kind = PromptKind::Text;
    std::string text_prompt;
    std::vector<std::uint32_t> token_prompt;
    std::vector<text::ChatMessage> messages;
    std::uint32_t maximum_tokens = 0;
    bool maximum_tokens_explicit = false;
    bool stream = false;
    bool enable_thinking = true;
    std::vector<std::uint32_t> stop_token_ids;
};

struct ParsedCompletionResult {
    ParsedCompletionRequest request;
    CompletionRequestError error = CompletionRequestError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == CompletionRequestError::None;
    }
};

struct PreparedCompletionRequest {
    Route route = Route::Completions;
    PromptKind prompt_kind = PromptKind::Text;
    std::vector<std::uint32_t> prompt_tokens;
    std::uint32_t maximum_tokens = 0;
    bool stream = false;
    bool enable_thinking = true;
    std::vector<std::uint32_t> stop_token_ids;
};

struct PreparedCompletionResult {
    PreparedCompletionRequest request;
    CompletionRequestError error = CompletionRequestError::None;
    std::string detail;

    explicit operator bool() const noexcept {
        return error == CompletionRequestError::None;
    }
};

ParsedCompletionResult parse_completion_request(
    Route route, const std::string& body, std::string_view model_package_id,
    const model::qwen36::TokenizerSpec& tokenizer_specification,
    const CompletionRequestLimits& limits);

PreparedCompletionResult prepare_completion_request(
    ParsedCompletionRequest request, const text::Tokenizer& tokenizer,
    const text::Qwen36ChatTemplate& chat_template,
    const CompletionRequestLimits& limits);

int completion_request_http_status(CompletionRequestError error) noexcept;
std::string_view completion_request_code(CompletionRequestError error) noexcept;

} // namespace tatara::service
