#include "tatara/service/server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace {

using namespace tatara::service;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

Configuration configuration() {
    Configuration value;
    value.model.record = "/tmp/record";
    value.model.artifact_root = "/tmp/root";
    // Port 0 asks the kernel for a free port, so tests never collide with a
    // real service or with each other.
    value.service.port = 0;
    return value;
}

ServiceSnapshot ready_snapshot() {
    ServiceSnapshot value;
    value.readiness = Readiness::Ready;
    value.model_package_id = "qwen36-35b-a3b";
    return value;
}

int connect_to(std::uint16_t port) {
    const int descriptor = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(descriptor, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(descriptor);
        return -1;
    }
    return descriptor;
}

std::string round_trip(std::uint16_t port, const std::string& request, int read_rounds = 1) {
    const int descriptor = connect_to(port);
    if (descriptor < 0) {
        return {};
    }
    ::send(descriptor, request.data(), request.size(), 0);
    std::string received;
    for (int round = 0; round < read_rounds; ++round) {
        char buffer[8192];
        const auto count = ::recv(descriptor, buffer, sizeof(buffer), 0);
        if (count <= 0) {
            break;
        }
        received.append(buffer, static_cast<std::size_t>(count));
    }
    ::close(descriptor);
    return received;
}

std::string get(std::string_view path) {
    std::string out("GET ");
    out.append(path).append(" HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n");
    return out;
}

std::string post(std::string_view path, std::string_view body) {
    std::string out("POST ");
    out.append(path).append(
        " HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\ncontent-length: ");
    out.append(std::to_string(body.size())).append("\r\n\r\n").append(body);
    return out;
}

struct Harness {
    Server server;
    std::thread thread;

    explicit Harness(ServerHooks hooks) : server(configuration(), std::move(hooks)) {}
    Harness(Configuration service_configuration, ServerHooks hooks)
        : server(std::move(service_configuration), std::move(hooks)) {}

    bool start() {
        if (!server.start()) {
            return false;
        }
        thread = std::thread([this] { server.run(); });
        // The listener is bound before run() starts, so a connect cannot race.
        return true;
    }

    void shutdown() {
        server.stop();
        if (thread.joinable()) {
            thread.join();
        }
    }
};

void operational_endpoints_answer() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    Harness harness(hooks);
    check(harness.start(), "server binds on an ephemeral port");

    const auto live = round_trip(harness.server.port(), get("/health/live"));
    const auto ready = round_trip(harness.server.port(), get("/health/ready"));
    const auto metrics = round_trip(harness.server.port(), get("/metrics"));
    const auto models = round_trip(harness.server.port(), get("/v1/models"));

    check(live.find("200 OK") != std::string::npos, "liveness answers 200");
    check(ready.find("200 OK") != std::string::npos, "readiness answers 200");
    check(metrics.find("tatara_ready 1") != std::string::npos, "metrics are scrapeable");
    check(models.find("qwen36-35b-a3b") != std::string::npos, "models names the model");
    harness.shutdown();
}

void an_unready_service_refuses_completions_but_stays_live() {
    ServerHooks hooks;
    hooks.snapshot = [] {
        ServiceSnapshot value;
        value.readiness = Readiness::ModelLoading;
        return value;
    };
    hooks.completions = [](Route, const std::string&, Generation&) { return true; };
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto live = round_trip(harness.server.port(), get("/health/live"));
    const auto completion =
        round_trip(harness.server.port(), post("/v1/completions", "{\"prompt\":\"x\"}"));

    check(live.find("200 OK") != std::string::npos, "live while loading");
    check(completion.find("503") != std::string::npos, "completions refused with 503");
    check(completion.find("tatara.not_ready") != std::string::npos, "with a stable code");
    harness.shutdown();
}

void a_non_streaming_completion_returns_a_body() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    hooks.completions = [](Route, const std::string&, Generation& generation) {
        generation.body = "{\"choices\":[{\"text\":\"hello\"}]}";
        return true;
    };
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto response =
        round_trip(harness.server.port(), post("/v1/completions", "{\"prompt\":\"x\"}"));

    check(response.find("200 OK") != std::string::npos, "200");
    check(response.find("\"text\":\"hello\"") != std::string::npos, "body is delivered");
    harness.shutdown();
}

void a_typed_request_failure_reaches_the_client() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    hooks.completions = [](Route route, const std::string&,
                           Generation& generation) {
        check(route == Route::ChatCompletions,
              "completion handler receives the resolved route");
        generation.response_status = 400;
        generation.body =
            "{\"error\":{\"code\":\"tatara.wrong_field_type\"}}";
        return false;
    };
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto response =
        round_trip(harness.server.port(),
                   post("/v1/chat/completions", "{}"));

    check(response.find("400 Bad Request") != std::string::npos,
          "typed handler status is preserved");
    check(response.find("tatara.wrong_field_type") != std::string::npos,
          "typed handler body is preserved");
    harness.shutdown();
}

void a_streaming_completion_emits_events_and_a_terminator() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    hooks.completions = [](Route, const std::string&, Generation& generation) {
        generation.stream = true;
        generation.emit("{\"delta\":\"a\"}");
        generation.emit("{\"delta\":\"b\"}");
        return true;
    };
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto response =
        round_trip(harness.server.port(), post("/v1/chat/completions", "{\"stream\":true}"), 6);

    check(response.find("text/event-stream") != std::string::npos, "SSE content type");
    check(response.find("data: {\"delta\":\"a\"}") != std::string::npos, "first token streamed");
    check(response.find("data: {\"delta\":\"b\"}") != std::string::npos, "second token streamed");
    check(response.find("data: [DONE]") != std::string::npos, "terminator sent");
    harness.shutdown();
}

// A client that disconnects must stop the work, not merely be ignored: paying
// for a response nobody reads is the failure mode cancellation exists to stop.
void a_disconnected_client_cancels_the_generation() {
    std::atomic<int> tokens_emitted{0};
    std::atomic<bool> observed_cancel{false};
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    hooks.completions = [&](Route, const std::string&, Generation& generation) {
        generation.stream = true;
        for (int index = 0; index < 200; ++index) {
            if (generation.cancelled()) {
                observed_cancel.store(true);
                return true;
            }
            generation.emit("{\"delta\":\"x\"}");
            tokens_emitted.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return true;
    };
    Harness harness(hooks);
    check(harness.start(), "starts");

    const int descriptor = connect_to(harness.server.port());
    const auto request = post("/v1/chat/completions", "{\"stream\":true}");
    ::send(descriptor, request.data(), request.size(), 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    ::close(descriptor);

    for (int wait = 0; wait < 200 && !observed_cancel.load(); ++wait) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    check(observed_cancel.load(), "the handler observed the client going away");
    check(tokens_emitted.load() < 200,
          "generation stopped early rather than running to completion");
    harness.shutdown();
}

void an_unknown_path_is_404_over_the_wire() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto response = round_trip(harness.server.port(), get("/v1/embeddings"));

    check(response.find("404") != std::string::npos, "unknown path is 404");
    harness.shutdown();
}

void a_malformed_request_is_rejected_and_closed() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    Harness harness(hooks);
    check(harness.start(), "starts");

    const auto response = round_trip(harness.server.port(), "GARBAGE\r\n\r\n");

    check(response.find("400") != std::string::npos, "malformed request is 400");
    harness.shutdown();
}

// Open sockets consume memory even before they become requests. Admission
// therefore bounds accepted connections, not only completed HTTP messages.
void accepted_connections_are_bounded() {
    auto limits = configuration();
    limits.service.max_concurrent_requests = 1;
    limits.service.queue_depth = 0;
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    Harness harness(std::move(limits), hooks);
    check(harness.start(), "starts");

    const int held = connect_to(harness.server.port());
    const std::string incomplete = "GET /health/live HTTP/1.1\r\n";
    ::send(held, incomplete.data(), incomplete.size(), 0);

    const int waiting = connect_to(harness.server.port());
    const auto request = get("/health/live");
    ::send(waiting, request.data(), request.size(), 0);

    pollfd descriptor{waiting, POLLIN, 0};
    check(::poll(&descriptor, 1, 100) == 0,
          "a second connection is not accepted while the one-slot bound is occupied");

    ::close(held);
    descriptor.revents = 0;
    check(::poll(&descriptor, 1, 2000) == 1 && (descriptor.revents & POLLIN) != 0,
          "the waiting connection is admitted after the slot is released");
    char response[1024];
    const auto count = ::recv(waiting, response, sizeof(response), 0);
    check(count > 0 && std::string_view(response, static_cast<std::size_t>(count)).find("200 OK") !=
                           std::string_view::npos,
          "the admitted connection is served");
    ::close(waiting);
    harness.shutdown();
}

// Drain stops accepting but must let the loop finish and exit cleanly rather
// than hanging or dropping the listener mid-flight.
void drain_stops_accepting_and_returns() {
    ServerHooks hooks;
    hooks.snapshot = ready_snapshot;
    Harness harness(hooks);
    check(harness.start(), "starts");
    const std::uint16_t port = harness.server.port();

    check(round_trip(port, get("/health/live")).find("200") != std::string::npos,
          "serves before drain");
    harness.server.stop();
    if (harness.thread.joinable()) {
        harness.thread.join();
    }

    check(connect_to(port) < 0 || round_trip(port, get("/health/live")).empty(),
          "after drain the listener no longer serves");
}

} // namespace

int main() {
    operational_endpoints_answer();
    an_unready_service_refuses_completions_but_stays_live();
    a_non_streaming_completion_returns_a_body();
    a_typed_request_failure_reaches_the_client();
    a_streaming_completion_emits_events_and_a_terminator();
    a_disconnected_client_cancels_the_generation();
    an_unknown_path_is_404_over_the_wire();
    a_malformed_request_is_rejected_and_closed();
    accepted_connections_are_bounded();
    drain_stops_accepting_and_returns();
    if (failures == 0) {
        std::printf("server: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
