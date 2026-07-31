#include "tatara/model/sha256.h"
#include "tatara/text/chat_template.h"
#include "tatara/text/tokenizer.h"

#include "tatara/generated/model_plan.h"

#import <Foundation/Foundation.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

using tatara::model::Sha256;
using tatara::model::qwen36::TokenizerSpec;
using tatara::model::qwen36::generated::kModelPlan;
using tatara::model::sha256_hex;
using tatara::text::ChatMessage;
using tatara::text::ChatTemplateError;
using tatara::text::ChatTemplateLoadError;
using tatara::text::Qwen36ChatTemplate;
using tatara::text::Tokenizer;
using tatara::text::TokenizerLimits;

void check(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::string sha256(std::string_view value) {
    Sha256 hasher;
    hasher.update(std::as_bytes(std::span(value.data(), value.size())));
    return sha256_hex(hasher.finish());
}

std::string std_string(NSString* value) {
    NSData* data =
        [value dataUsingEncoding:NSUTF8StringEncoding allowLossyConversion:NO];
    return std::string(static_cast<const char*>(data.bytes), data.length);
}

class TemporaryDirectory {
  public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path() /
                ("tatara-template-" + std::to_string(::getpid()))) {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directory(path_);
    }

    ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

    const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

void synthetic_contract() {
    TemporaryDirectory directory;
    const std::string source = "synthetic compiled-template identity";
    {
        std::ofstream stream(directory.path() / "chat_template.jinja",
                             std::ios::binary);
        stream.write(source.data(), static_cast<std::streamsize>(source.size()));
    }
    const std::string digest = sha256(source);
    TokenizerSpec specification = kModelPlan.tokenizer;
    specification.template_sha256 = digest;
    specification.template_size_bytes = source.size();

    auto loaded = Qwen36ChatTemplate::load(specification, directory.path());
    check(static_cast<bool>(loaded), loaded.detail);

    const std::vector<ChatMessage> simple{
        ChatMessage{.role = "user", .content = " hi "},
    };
    auto thinking = loaded.chat_template->render(simple, true, 1024);
    check(static_cast<bool>(thinking), thinking.detail);
    check(thinking.prompt ==
              "<|im_start|>user\nhi<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n",
          "simple thinking prompt");
    auto direct = loaded.chat_template->render(simple, false, 1024);
    check(static_cast<bool>(direct), direct.detail);
    check(direct.prompt.ends_with("<think>\n\n</think>\n\n"),
          "thinking-disabled prompt");

    const std::vector<ChatMessage> history{
        ChatMessage{.role = "system", .content = " system "},
        ChatMessage{.role = "user", .content = "q1"},
        ChatMessage{
            .role = "assistant",
            .content = "visible",
            .reasoning_content = " reason ",
        },
        ChatMessage{.role = "user", .content = "q2"},
    };
    auto rendered_history = loaded.chat_template->render(history, true, 4096);
    check(static_cast<bool>(rendered_history), rendered_history.detail);
    check(rendered_history.prompt.find(
              "<|im_start|>assistant\nvisible<|im_end|>") !=
              std::string::npos,
          "reasoning before the latest query is suppressed");

    const std::vector<ChatMessage> bad_role{
        ChatMessage{.role = "user", .content = "q"},
        ChatMessage{.role = "tool", .content = "r"},
    };
    check(loaded.chat_template->render(bad_role, true, 1024).error ==
              ChatTemplateError::UnsupportedRole,
          "unsupported role is typed");
    check(loaded.chat_template->render({}, true, 1024).error ==
              ChatTemplateError::EmptyMessages,
          "empty messages are typed");
    check(loaded.chat_template->render(simple, true, 8).error ==
              ChatTemplateError::OutputLimitExceeded,
          "output bound is typed");

    TokenizerSpec wrong_digest = specification;
    wrong_digest.template_sha256 = std::string_view{
        "0000000000000000000000000000000000000000000000000000000000000000"};
    check(Qwen36ChatTemplate::load(wrong_digest, directory.path()).error ==
              ChatTemplateLoadError::DigestMismatch,
          "template digest disagreement is typed");
}

void exact_artifact_parity(const std::filesystem::path& model_root,
                           const std::filesystem::path& fixture_path) {
    auto chat = Qwen36ChatTemplate::load(kModelPlan.tokenizer, model_root);
    check(static_cast<bool>(chat), chat.detail);
    auto tokenizer = Tokenizer::load(kModelPlan.tokenizer, model_root);
    check(static_cast<bool>(tokenizer), tokenizer.detail);

    std::ifstream fixture(fixture_path);
    check(static_cast<bool>(fixture), "template fixture opens");
    std::string line;
    std::size_t rows = 0;
    while (std::getline(fixture, line)) {
        if (line.empty()) {
            continue;
        }
        @autoreleasepool {
            NSData* data = [NSData dataWithBytes:line.data() length:line.size()];
            NSError* error = nil;
            id parsed =
                [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
            check(parsed != nil && [parsed isKindOfClass:NSDictionary.class],
                  "template fixture row is valid JSON");
            NSDictionary* row = static_cast<NSDictionary*>(parsed);
            NSArray* raw_messages = row[@"messages"];
            NSString* expected_prompt = row[@"prompt"];
            NSArray* expected_ids = row[@"ids"];
            NSNumber* enable_thinking = row[@"enable_thinking"];
            check([raw_messages isKindOfClass:NSArray.class] &&
                      [expected_prompt isKindOfClass:NSString.class] &&
                      [expected_ids isKindOfClass:NSArray.class] &&
                      [enable_thinking isKindOfClass:NSNumber.class],
                  "template fixture row schema");
            std::vector<ChatMessage> messages;
            messages.reserve(raw_messages.count);
            for (NSDictionary* raw_message in raw_messages) {
                check([raw_message isKindOfClass:NSDictionary.class],
                      "template message schema");
                ChatMessage message{
                    .role = std_string(raw_message[@"role"]),
                    .content = std_string(raw_message[@"content"]),
                    .reasoning_content = std::nullopt,
                };
                if (raw_message[@"reasoning_content"] != nil) {
                    message.reasoning_content =
                        std_string(raw_message[@"reasoning_content"]);
                }
                messages.push_back(std::move(message));
            }
            auto rendered = chat.chat_template->render(
                messages, enable_thinking.boolValue, 1U << 20U);
            check(static_cast<bool>(rendered), rendered.detail);
            check(rendered.prompt == std_string(expected_prompt),
                  "template bytes parity row " + std::to_string(rows));

            std::vector<std::uint32_t> ids;
            ids.reserve(expected_ids.count);
            for (NSNumber* identifier in expected_ids) {
                ids.push_back(identifier.unsignedIntValue);
            }
            auto encoded = tokenizer.tokenizer->encode(
                rendered.prompt,
                TokenizerLimits{
                    .maximum_input_bytes = 1U << 20U,
                    .maximum_tokens = 1U << 18U,
                    .maximum_output_bytes = 1U << 20U,
                });
            check(static_cast<bool>(encoded), encoded.detail);
            check(encoded.tokens == ids,
                  "template token parity row " + std::to_string(rows));
            ++rows;
        }
    }
    check(rows > 0, "template fixture is non-empty");
    std::cout << "real chat-template parity " << rows << '/' << rows
              << " prompt+ids exact\n";
}

} // namespace

int main(int argument_count, char** arguments) {
    @autoreleasepool {
        synthetic_contract();
        if (argument_count == 3) {
            exact_artifact_parity(arguments[1], arguments[2]);
        } else {
            check(argument_count == 1,
                  "usage: chat_template_test [model-root parity-fixture]");
        }
    }
    return 0;
}
