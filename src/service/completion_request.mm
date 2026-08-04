#include "tatara/service/completion_request.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tatara::service {
namespace {

ParsedCompletionResult parse_error(CompletionRequestError error,
                                   std::string detail) {
    return {
        .request = {},
        .error = error,
        .detail = std::move(detail),
    };
}

PreparedCompletionResult prepare_error(CompletionRequestError error,
                                       std::string detail) {
    return {
        .request = {},
        .error = error,
        .detail = std::move(detail),
    };
}

std::string std_string(NSString* value) {
    if (value == nil) {
        return {};
    }
    NSData* data =
        [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    if (data == nil) {
        return {};
    }
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

bool is_boolean(id value) {
    return [value isKindOfClass:NSNumber.class] &&
           CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

bool unsigned_integer(id value, std::uint32_t& result) {
    if (![value isKindOfClass:NSNumber.class] || is_boolean(value)) {
        return false;
    }
    NSNumber* number = static_cast<NSNumber*>(value);
    const long long signed_value = number.longLongValue;
    const unsigned long long unsigned_value = number.unsignedLongLongValue;
    if (signed_value < 0 ||
        unsigned_value > std::numeric_limits<std::uint32_t>::max() ||
        number.doubleValue != static_cast<double>(unsigned_value)) {
        return false;
    }
    result = static_cast<std::uint32_t>(unsigned_value);
    return true;
}

bool allowed_key(NSString* key, const std::vector<std::string_view>& allowed) {
    const std::string value = std_string(key);
    return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

std::optional<std::string> first_unknown_key(
    NSDictionary* object, const std::vector<std::string_view>& allowed) {
    for (id key in object) {
        if (![key isKindOfClass:NSString.class] ||
            !allowed_key(static_cast<NSString*>(key), allowed)) {
            return std::optional<std::string>{
                [key isKindOfClass:NSString.class]
                    ? std_string(static_cast<NSString*>(key))
                    : "<non-string>"};
        }
    }
    return std::nullopt;
}

CompletionRequestError parse_common(
    NSDictionary* root, std::string_view model_package_id,
    const model::qwen36::TokenizerSpec& tokenizer_specification,
    const CompletionRequestLimits& limits, ParsedCompletionRequest& request,
    std::string& detail) {
    id model = root[@"model"];
    if (model == nil) {
        detail = "missing required field: model";
        return CompletionRequestError::MissingField;
    }
    if (![model isKindOfClass:NSString.class]) {
        detail = "model must be a string";
        return CompletionRequestError::WrongFieldType;
    }
    if (std_string(static_cast<NSString*>(model)) != model_package_id) {
        detail = "request model does not match the loaded package";
        return CompletionRequestError::WrongModel;
    }

    request.maximum_tokens = limits.default_maximum_tokens;
    request.maximum_tokens_explicit = false;
    if (id maximum_tokens = root[@"max_tokens"]; maximum_tokens != nil) {
        if (!unsigned_integer(maximum_tokens, request.maximum_tokens)) {
            detail = "max_tokens must be a non-negative uint32";
            return CompletionRequestError::WrongFieldType;
        }
        request.maximum_tokens_explicit = true;
    }
    if (request.maximum_tokens == 0) {
        detail = "max_tokens must be at least 1";
        return CompletionRequestError::InvalidMaximumTokens;
    }

    request.stream = false;
    if (id stream = root[@"stream"]; stream != nil) {
        if (!is_boolean(stream)) {
            detail = "stream must be a boolean";
            return CompletionRequestError::WrongFieldType;
        }
        request.stream = [static_cast<NSNumber*>(stream) boolValue];
    }

    request.stop_token_ids.clear();
    if (id raw_stops = root[@"stop_token_ids"]; raw_stops != nil) {
        if (![raw_stops isKindOfClass:NSArray.class]) {
            detail = "stop_token_ids must be an array";
            return CompletionRequestError::WrongFieldType;
        }
        NSArray* stops = static_cast<NSArray*>(raw_stops);
        if (stops.count == 0 ||
            stops.count > model::qwen36::kMaximumStopTokens) {
            detail = "stop_token_ids is empty or exceeds the package bound";
            return CompletionRequestError::UnsupportedFeature;
        }
        request.stop_token_ids.reserve(stops.count);
        for (id raw_stop in stops) {
            std::uint32_t stop = 0;
            if (!unsigned_integer(raw_stop, stop)) {
                detail = "stop_token_ids entries must be uint32 values";
                return CompletionRequestError::WrongFieldType;
            }
            if (stop >= tokenizer_specification.populated_vocabulary ||
                std::find(request.stop_token_ids.begin(),
                          request.stop_token_ids.end(),
                          stop) != request.stop_token_ids.end()) {
                detail = "stop token is unpopulated or repeated";
                return CompletionRequestError::InvalidTokenId;
            }
            request.stop_token_ids.push_back(stop);
        }
    } else {
        request.stop_token_ids.assign(
            tokenizer_specification.stop_token_ids.begin(),
            tokenizer_specification.stop_token_ids.begin() +
                tokenizer_specification.stop_token_count);
    }
    return CompletionRequestError::None;
}

CompletionRequestError parse_completion_prompt(
    NSDictionary* root, const model::qwen36::TokenizerSpec& specification,
    const CompletionRequestLimits& limits, ParsedCompletionRequest& request,
    std::string& detail) {
    id prompt = root[@"prompt"];
    if (prompt == nil) {
        detail = "missing required field: prompt";
        return CompletionRequestError::MissingField;
    }
    if ([prompt isKindOfClass:NSString.class]) {
        request.prompt_kind = PromptKind::Text;
        request.text_prompt = std_string(static_cast<NSString*>(prompt));
        if (request.text_prompt.empty()) {
            detail = "text prompt must be non-empty";
            return CompletionRequestError::EmptyPrompt;
        }
        if (request.text_prompt.size() > limits.maximum_input_bytes) {
            detail = "text prompt exceeds the configured byte bound";
            return CompletionRequestError::BodyLimitExceeded;
        }
        return CompletionRequestError::None;
    }
    if (![prompt isKindOfClass:NSArray.class]) {
        detail = "prompt must be a string or token-ID array";
        return CompletionRequestError::WrongFieldType;
    }
    NSArray* identifiers = static_cast<NSArray*>(prompt);
    if (identifiers.count == 0) {
        detail = "token prompt must be non-empty";
        return CompletionRequestError::EmptyPrompt;
    }
    if (identifiers.count > limits.maximum_prompt_tokens) {
        detail = "token prompt exceeds the configured token bound";
        return CompletionRequestError::ContextExceeded;
    }
    request.prompt_kind = PromptKind::TokenIds;
    request.token_prompt.reserve(identifiers.count);
    for (id raw_identifier in identifiers) {
        std::uint32_t identifier = 0;
        if (!unsigned_integer(raw_identifier, identifier)) {
            detail = "prompt token IDs must be uint32 values";
            return CompletionRequestError::WrongFieldType;
        }
        if (identifier >= specification.vocabulary) {
            detail = "prompt token ID exceeds the model vocabulary";
            return CompletionRequestError::InvalidTokenId;
        }
        request.token_prompt.push_back(identifier);
    }
    return CompletionRequestError::None;
}

CompletionRequestError parse_chat_messages(NSDictionary* root,
                                           ParsedCompletionRequest& request,
                                           std::string& detail) {
    id raw_messages = root[@"messages"];
    if (raw_messages == nil) {
        detail = "missing required field: messages";
        return CompletionRequestError::MissingField;
    }
    if (![raw_messages isKindOfClass:NSArray.class]) {
        detail = "messages must be an array";
        return CompletionRequestError::WrongFieldType;
    }
    NSArray* messages = static_cast<NSArray*>(raw_messages);
    if (messages.count == 0) {
        detail = "messages must be non-empty";
        return CompletionRequestError::EmptyPrompt;
    }
    request.prompt_kind = PromptKind::Chat;
    request.messages.reserve(messages.count);
    const std::vector<std::string_view> allowed_message_keys{
        "role", "content", "reasoning_content"};
    for (id raw_message in messages) {
        if (![raw_message isKindOfClass:NSDictionary.class]) {
            detail = "each message must be an object";
            return CompletionRequestError::WrongFieldType;
        }
        NSDictionary* message = static_cast<NSDictionary*>(raw_message);
        const std::optional<std::string> unknown =
            first_unknown_key(message, allowed_message_keys);
        if (unknown.has_value()) {
            detail = "unsupported message field: " + *unknown;
            return CompletionRequestError::UnsupportedFeature;
        }
        id role = message[@"role"];
        id content = message[@"content"];
        if (role == nil || content == nil) {
            detail = "each message requires role and content";
            return CompletionRequestError::MissingField;
        }
        if (![role isKindOfClass:NSString.class]) {
            detail = "message role and content must be strings";
            return CompletionRequestError::WrongFieldType;
        }
        if (![content isKindOfClass:NSString.class]) {
            detail = "structured message content is outside the text-only contract";
            return [content isKindOfClass:NSArray.class] ||
                           [content isKindOfClass:NSDictionary.class]
                       ? CompletionRequestError::UnsupportedFeature
                       : CompletionRequestError::WrongFieldType;
        }
        text::ChatMessage parsed{
            .role = std_string(static_cast<NSString*>(role)),
            .content = std_string(static_cast<NSString*>(content)),
            .reasoning_content = std::nullopt,
        };
        if (id reasoning = message[@"reasoning_content"]; reasoning != nil) {
            if (![reasoning isKindOfClass:NSString.class]) {
                detail = "reasoning_content must be a string";
                return CompletionRequestError::WrongFieldType;
            }
            parsed.reasoning_content =
                std_string(static_cast<NSString*>(reasoning));
        }
        request.messages.push_back(std::move(parsed));
    }
    return CompletionRequestError::None;
}

} // namespace

ParsedCompletionResult parse_completion_request(
    Route route, const std::string& body, std::string_view model_package_id,
    const model::qwen36::TokenizerSpec& tokenizer_specification,
    const CompletionRequestLimits& limits) {
    try {
        if (route != Route::Completions && route != Route::ChatCompletions) {
            return parse_error(CompletionRequestError::UnsupportedFeature,
                               "route is not a completion endpoint");
        }
        if (body.size() > limits.maximum_body_bytes) {
            return parse_error(CompletionRequestError::BodyLimitExceeded,
                               "request body exceeds the configured byte bound");
        }
        @autoreleasepool {
            NSData* data = [NSData dataWithBytes:body.data() length:body.size()];
            NSError* json_error = nil;
            id parsed =
                [NSJSONSerialization JSONObjectWithData:data options:0 error:&json_error];
            if (parsed == nil || ![parsed isKindOfClass:NSDictionary.class]) {
                return parse_error(
                    CompletionRequestError::InvalidJson,
                    json_error == nil ? "request JSON root must be an object"
                                      : std_string(json_error.localizedDescription));
            }
            NSDictionary* root = static_cast<NSDictionary*>(parsed);
            const std::vector<std::string_view> completion_keys{
                "model", "prompt", "max_tokens", "stream", "stop_token_ids"};
            const std::vector<std::string_view> chat_keys{
                "model", "messages", "max_tokens", "stream",
                "stop_token_ids", "enable_thinking", "tools"};
            const std::optional<std::string> unknown = first_unknown_key(
                root, route == Route::Completions ? completion_keys : chat_keys);
            if (unknown.has_value()) {
                return parse_error(CompletionRequestError::UnknownField,
                                   "unsupported request field: " + *unknown);
            }
            // Clients commonly send an empty tools array whether or
            // not they use tools; a populated one is refused because
            // this engine cannot call tools.
            if (route == Route::ChatCompletions) {
                if (id tools = root[@"tools"]; tools != nil &&
                                               tools != [NSNull null]) {
                    if (![tools isKindOfClass:[NSArray class]]) {
                        return parse_error(
                            CompletionRequestError::WrongFieldType,
                            "tools must be an array");
                    }
                    if ([static_cast<NSArray*>(tools) count] != 0) {
                        return parse_error(
                            CompletionRequestError::UnknownField,
                            "tool calling is not supported by this engine;"
                            " send tools omitted or empty");
                    }
                }
            }

            ParsedCompletionRequest request;
            request.route = route;
            request.enable_thinking = tokenizer_specification.default_thinking;
            std::string detail;
            if (const CompletionRequestError error =
                    parse_common(root, model_package_id, tokenizer_specification,
                                 limits, request, detail);
                error != CompletionRequestError::None) {
                return parse_error(error, std::move(detail));
            }
            if (route == Route::Completions) {
                if (const CompletionRequestError error =
                        parse_completion_prompt(root, tokenizer_specification,
                                                limits, request, detail);
                    error != CompletionRequestError::None) {
                    return parse_error(error, std::move(detail));
                }
            } else {
                if (id thinking = root[@"enable_thinking"]; thinking != nil) {
                    if (!is_boolean(thinking)) {
                        return parse_error(
                            CompletionRequestError::WrongFieldType,
                            "enable_thinking must be a boolean");
                    }
                    request.enable_thinking =
                        [static_cast<NSNumber*>(thinking) boolValue];
                }
                if (const CompletionRequestError error =
                        parse_chat_messages(root, request, detail);
                    error != CompletionRequestError::None) {
                    return parse_error(error, std::move(detail));
                }
            }
            return {
                .request = std::move(request),
                .error = CompletionRequestError::None,
                .detail = {},
            };
        }
    } catch (const std::bad_alloc&) {
        return parse_error(CompletionRequestError::AllocationFailure,
                           "request parsing allocation failed");
    }
}

PreparedCompletionResult prepare_completion_request(
    ParsedCompletionRequest request, const text::Tokenizer& tokenizer,
    const text::Qwen36ChatTemplate& chat_template,
    const CompletionRequestLimits& limits) {
    try {
        PreparedCompletionRequest prepared{
            .route = request.route,
            .prompt_kind = request.prompt_kind,
            .prompt_tokens = {},
            .maximum_tokens = request.maximum_tokens,
            .stream = request.stream,
            .enable_thinking = request.enable_thinking,
            .stop_token_ids = std::move(request.stop_token_ids),
        };
        if (request.prompt_kind == PromptKind::TokenIds) {
            prepared.prompt_tokens = std::move(request.token_prompt);
        } else {
            std::string prompt;
            if (request.prompt_kind == PromptKind::Text) {
                prompt = std::move(request.text_prompt);
            } else {
                text::ChatRenderResult rendered = chat_template.render(
                    request.messages, request.enable_thinking,
                    limits.maximum_input_bytes);
                if (!rendered) {
                    return prepare_error(CompletionRequestError::TemplateFailure,
                                         std::move(rendered.detail));
                }
                prompt = std::move(rendered.prompt);
            }
            text::EncodeResult encoded = tokenizer.encode(
                prompt,
                text::TokenizerLimits{
                    .maximum_input_bytes = limits.maximum_input_bytes,
                    .maximum_tokens = limits.maximum_prompt_tokens,
                    .maximum_output_bytes = limits.maximum_input_bytes,
                });
            if (!encoded) {
                const CompletionRequestError error =
                    encoded.error == text::TokenizerError::TokenLimitExceeded
                        ? CompletionRequestError::ContextExceeded
                        : CompletionRequestError::TokenizerFailure;
                return prepare_error(error, std::move(encoded.detail));
            }
            prepared.prompt_tokens = std::move(encoded.tokens);
        }

        if (prepared.prompt_tokens.empty()) {
            return prepare_error(CompletionRequestError::EmptyPrompt,
                                 "prepared prompt contains no tokens");
        }
        if (prepared.prompt_tokens.size() > limits.maximum_prompt_tokens ||
            prepared.prompt_tokens.size() >=
                limits.maximum_context_tokens) {
            return prepare_error(
                CompletionRequestError::ContextExceeded,
                "prompt plus requested output exceeds the configured context");
        }
        const std::uint32_t available =
            limits.maximum_context_tokens -
            static_cast<std::uint32_t>(prepared.prompt_tokens.size());
        if (request.maximum_tokens_explicit &&
            prepared.maximum_tokens > available) {
            return prepare_error(
                CompletionRequestError::ContextExceeded,
                "prompt plus requested output exceeds the configured context");
        }
        if (!request.maximum_tokens_explicit) {
            prepared.maximum_tokens =
                std::min(prepared.maximum_tokens, available);
        }
        return {
            .request = std::move(prepared),
            .error = CompletionRequestError::None,
            .detail = {},
        };
    } catch (const std::bad_alloc&) {
        return prepare_error(CompletionRequestError::AllocationFailure,
                             "request preparation allocation failed");
    }
}

int completion_request_http_status(CompletionRequestError error) noexcept {
    switch (error) {
    case CompletionRequestError::BodyLimitExceeded:
        return 413;
    case CompletionRequestError::ContextExceeded:
        return 400;
    case CompletionRequestError::AllocationFailure:
        return 500;
    case CompletionRequestError::None:
        return 200;
    default:
        return 400;
    }
}

std::string_view completion_request_code(CompletionRequestError error) noexcept {
    switch (error) {
    case CompletionRequestError::None:
        return "";
    case CompletionRequestError::BodyLimitExceeded:
        return "tatara.body_limit";
    case CompletionRequestError::InvalidJson:
        return "tatara.invalid_json";
    case CompletionRequestError::MissingField:
        return "tatara.missing_field";
    case CompletionRequestError::UnknownField:
        return "tatara.unknown_field";
    case CompletionRequestError::WrongFieldType:
        return "tatara.wrong_field_type";
    case CompletionRequestError::WrongModel:
        return "tatara.wrong_model";
    case CompletionRequestError::UnsupportedFeature:
        return "tatara.unsupported_feature";
    case CompletionRequestError::EmptyPrompt:
        return "tatara.empty_prompt";
    case CompletionRequestError::InvalidTokenId:
        return "tatara.invalid_token";
    case CompletionRequestError::InvalidMaximumTokens:
        return "tatara.invalid_max_tokens";
    case CompletionRequestError::ContextExceeded:
        return "tatara.context_exceeded";
    case CompletionRequestError::TokenizerFailure:
        return "tatara.tokenizer_failure";
    case CompletionRequestError::TemplateFailure:
        return "tatara.template_failure";
    case CompletionRequestError::AllocationFailure:
        return "tatara.allocation_failure";
    }
    return "tatara.invalid_request";
}

} // namespace tatara::service
