#include "tatara/service/completion_request.h"

#include "tatara/generated/model_plan.h"

#import <Foundation/Foundation.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tatara::model::qwen36::generated::kModelPlan;
using tatara::service::CompletionRequestError;
using tatara::service::CompletionRequestLimits;
using tatara::service::PromptKind;
using tatara::service::Route;
using tatara::service::completion_request_code;
using tatara::service::completion_request_http_status;
using tatara::service::parse_completion_request;
using tatara::service::prepare_completion_request;
using tatara::text::Qwen36ChatTemplate;
using tatara::text::Tokenizer;

void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

CompletionRequestLimits limits() {
    return {
        .maximum_body_bytes = 1U << 20U,
        .maximum_input_bytes = 1U << 18U,
        .maximum_prompt_tokens = 16'000,
        .maximum_context_tokens = 16'384,
        .default_maximum_tokens = 16,
    };
}

void schema_contract() {
    const auto& specification = kModelPlan.tokenizer;
    const std::string model(kModelPlan.id);

    auto text = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model +
            "\",\"prompt\":\"hello\",\"max_tokens\":8,\"stream\":true}",
        model, specification, limits());
    check(static_cast<bool>(text), text.detail);
    check(text.request.prompt_kind == PromptKind::Text &&
              text.request.text_prompt == "hello" &&
              text.request.maximum_tokens == 8 &&
              text.request.maximum_tokens_explicit &&
              text.request.stream,
          "text completion schema");
    check(text.request.stop_token_ids ==
              std::vector<std::uint32_t>({248'046, 248'044}),
          "package default stop set");

    auto tokens = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model +
            "\",\"prompt\":[1,248319],\"stop_token_ids\":[248046]}",
        model, specification, limits());
    check(static_cast<bool>(tokens), tokens.detail);
    check(tokens.request.prompt_kind == PromptKind::TokenIds &&
              tokens.request.token_prompt ==
                  std::vector<std::uint32_t>({1, 248'319}) &&
              !tokens.request.maximum_tokens_explicit,
          "pretokenized prompt admits the reserved model tail");

    auto large_output = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model +
            "\",\"prompt\":[1],\"max_tokens\":200000}",
        model, specification, limits());
    check(static_cast<bool>(large_output) &&
              large_output.request.maximum_tokens == 200'000,
          "parser imposes no independent output-token ceiling");

    auto chat = parse_completion_request(
        Route::ChatCompletions,
        "{\"model\":\"" + model +
            "\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"enable_thinking\":false}",
        model, specification, limits());
    check(static_cast<bool>(chat), chat.detail);
    check(chat.request.prompt_kind == PromptKind::Chat &&
              !chat.request.enable_thinking && chat.request.messages.size() == 1,
          "chat text schema");

    check(parse_completion_request(Route::Completions, "{", model,
                                   specification, limits())
              .error == CompletionRequestError::InvalidJson,
          "invalid JSON is typed");
    check(parse_completion_request(
              Route::Completions,
              "{\"model\":\"" + model +
                  "\",\"prompt\":\"x\",\"temperature\":0}",
              model, specification, limits())
              .error == CompletionRequestError::UnknownField,
          "unimplemented sampling is rejected");
    check(parse_completion_request(
              Route::Completions,
              "{\"model\":\"wrong\",\"prompt\":\"x\"}",
              model, specification, limits())
              .error == CompletionRequestError::WrongModel,
          "wrong model is typed");
    check(parse_completion_request(
              Route::Completions,
              "{\"model\":\"" + model + "\",\"prompt\":[]}",
              model, specification, limits())
              .error == CompletionRequestError::EmptyPrompt,
          "empty token prompt is typed");
    check(parse_completion_request(
              Route::Completions,
              "{\"model\":\"" + model + "\",\"prompt\":[248320]}",
              model, specification, limits())
              .error == CompletionRequestError::InvalidTokenId,
          "out-of-model prompt ID is rejected");
    check(parse_completion_request(
              Route::Completions,
              "{\"model\":\"" + model +
                  "\",\"prompt\":\"x\",\"stop_token_ids\":[248100]}",
              model, specification, limits())
              .error == CompletionRequestError::InvalidTokenId,
          "unpopulated text stop ID is rejected");
    check(parse_completion_request(
              Route::ChatCompletions,
              "{\"model\":\"" + model +
                  "\",\"messages\":[{\"role\":\"user\",\"content\":["
                  "{\"type\":\"text\",\"text\":\"x\"}]}]}",
              model, specification, limits())
              .error == CompletionRequestError::UnsupportedFeature,
          "structured chat content is explicitly unsupported");
    check(parse_completion_request(
              Route::ChatCompletions,
              "{\"model\":\"" + model +
                  "\",\"messages\":[{\"role\":\"user\",\"content\":\"x\","
                  "\"tool_calls\":[]}]}",
              model, specification, limits())
              .error == CompletionRequestError::UnsupportedFeature,
          "tool calls are explicitly unsupported");

    CompletionRequestLimits tiny = limits();
    tiny.maximum_body_bytes = 8;
    check(parse_completion_request(Route::Completions, "123456789", model,
                                   specification, tiny)
              .error == CompletionRequestError::BodyLimitExceeded,
          "body limit is checked before JSON parsing");
    check(completion_request_http_status(
              CompletionRequestError::ContextExceeded) == 400 &&
              completion_request_code(CompletionRequestError::WrongModel) ==
                  "tatara.wrong_model",
          "typed error maps to stable HTTP evidence");
}

void real_text_preparation(const std::filesystem::path& model_root) {
    auto tokenizer = Tokenizer::load(kModelPlan.tokenizer, model_root);
    check(static_cast<bool>(tokenizer), tokenizer.detail);
    auto chat = Qwen36ChatTemplate::load(kModelPlan.tokenizer, model_root);
    check(static_cast<bool>(chat), chat.detail);
    const std::string model(kModelPlan.id);

    auto raw = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model + "\",\"prompt\":\"hello\"}",
        model, kModelPlan.tokenizer, limits());
    check(static_cast<bool>(raw), raw.detail);
    auto prepared_raw = prepare_completion_request(
        std::move(raw.request), *tokenizer.tokenizer, *chat.chat_template,
        limits());
    check(static_cast<bool>(prepared_raw), prepared_raw.detail);

    std::string token_body =
        "{\"model\":\"" + model + "\",\"prompt\":[";
    for (std::size_t index = 0;
         index < prepared_raw.request.prompt_tokens.size(); ++index) {
        token_body += std::to_string(prepared_raw.request.prompt_tokens[index]);
        token_body +=
            index + 1 == prepared_raw.request.prompt_tokens.size() ? "]}" : ",";
    }
    auto token_request = parse_completion_request(
        Route::Completions, token_body, model, kModelPlan.tokenizer, limits());
    auto prepared_tokens = prepare_completion_request(
        std::move(token_request.request), *tokenizer.tokenizer,
        *chat.chat_template, limits());
    check(static_cast<bool>(prepared_tokens), prepared_tokens.detail);
    check(prepared_raw.request.prompt_tokens ==
              prepared_tokens.request.prompt_tokens,
          "raw and pretokenized completion prompts are equivalent");

    auto chat_request = parse_completion_request(
        Route::ChatCompletions,
        "{\"model\":\"" + model +
            "\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
            "\"enable_thinking\":false}",
        model, kModelPlan.tokenizer, limits());
    auto prepared_chat = prepare_completion_request(
        std::move(chat_request.request), *tokenizer.tokenizer,
        *chat.chat_template, limits());
    check(static_cast<bool>(prepared_chat), prepared_chat.detail);
    check(!prepared_chat.request.prompt_tokens.empty(),
          "chat request renders and tokenizes");

    auto full_remaining = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model +
            "\",\"prompt\":[1],\"max_tokens\":16383}",
        model, kModelPlan.tokenizer, limits());
    auto full_remaining_prepared = prepare_completion_request(
        std::move(full_remaining.request), *tokenizer.tokenizer,
        *chat.chat_template, limits());
    check(static_cast<bool>(full_remaining_prepared) &&
              full_remaining_prepared.request.maximum_tokens == 16'383,
          "explicit output uses the entire remaining context");

    auto one_past = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model +
            "\",\"prompt\":[1],\"max_tokens\":16384}",
        model, kModelPlan.tokenizer, limits());
    auto one_past_prepared = prepare_completion_request(
        std::move(one_past.request), *tokenizer.tokenizer,
        *chat.chat_template, limits());
    check(one_past_prepared.error ==
              CompletionRequestError::ContextExceeded,
          "explicit output one past remaining context is rejected");

    CompletionRequestLimits clamped_default = limits();
    clamped_default.maximum_context_tokens =
        static_cast<std::uint32_t>(
            prepared_raw.request.prompt_tokens.size()) +
        4u;
    clamped_default.maximum_prompt_tokens =
        clamped_default.maximum_context_tokens - 1u;
    auto default_request = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model + "\",\"prompt\":\"hello\"}",
        model, kModelPlan.tokenizer, clamped_default);
    auto default_prepared = prepare_completion_request(
        std::move(default_request.request), *tokenizer.tokenizer,
        *chat.chat_template, clamped_default);
    check(static_cast<bool>(default_prepared) &&
              default_prepared.request.maximum_tokens == 4,
          "omitted output convenience default clamps to remaining context");

    CompletionRequestLimits no_room = limits();
    no_room.maximum_context_tokens =
        prepared_raw.request.maximum_tokens;
    auto repeated = parse_completion_request(
        Route::Completions,
        "{\"model\":\"" + model + "\",\"prompt\":\"hello\"}",
        model, kModelPlan.tokenizer, no_room);
    auto rejected = prepare_completion_request(
        std::move(repeated.request), *tokenizer.tokenizer,
        *chat.chat_template, no_room);
    check(rejected.error == CompletionRequestError::ContextExceeded,
          "context is rejected before execution");

    std::cout << "real completion preparation PASS: raw/token IDs equivalent, "
                 "chat admitted, context refusal typed\n";
}

} // namespace

int main(int argument_count, char** arguments) {
    @autoreleasepool {
        schema_contract();
        if (argument_count == 2) {
            real_text_preparation(arguments[1]);
        } else {
            check(argument_count == 1,
                  "usage: completion_request_test [model-root]");
        }
    }
    return 0;
}
