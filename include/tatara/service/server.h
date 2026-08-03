#pragma once

#include "tatara/service/admission.h"
#include "tatara/service/configuration.h"
#include "tatara/service/observability.h"
#include "tatara/service/router.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

namespace tatara::service {

// A generation in progress. The handler emits tokens through `emit` and must
// consult `cancelled` between them, so a client that disconnects stops work
// rather than paying for a response nobody will read.
struct Generation {
    std::string body;
    bool stream{false};
    int response_status{200};
    std::string response_content_type{"application/json"};
    std::function<bool(std::string_view)> emit;
    std::function<bool()> cancelled;
};

// Produces a completion. Returning false means the request failed and the
// server should answer 500. Kept as a callback so the transport is testable
// without the engine, and the engine testable without a socket.
using CompletionHandler =
    std::function<bool(Route route, const std::string& request_body, Generation&)>;

struct ServerHooks {
    CompletionHandler completions;
    std::function<ServiceSnapshot()> snapshot;
};

class Server {
  public:
    Server(Configuration configuration, ServerHooks hooks);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    // Binds and listens. Returns false if the address is unavailable.
    bool start();

    // Serves until `stop()` is called, then drains: stops accepting, finishes
    // in-flight connections, and returns. Blocks the calling thread.
    void run();

    // Safe from a signal handler.
    void stop();

    std::uint16_t port() const;

  private:
    // Returns whether the connection should stay open.
    bool handle(int descriptor, const struct HttpRequest& request);

    // Worker-thread path for completion routes when the configuration
    // admits more than one concurrent request: the worker owns the
    // socket, serves the parsed request, then continues the connection's
    // keep-alive loop off the poll thread.
    void serve_on_worker(int descriptor, struct HttpRequest first);

    Configuration configuration_;
    ServerHooks hooks_;
    int listener_{-1};
    std::uint16_t port_{0};
    std::atomic<bool> draining_{false};
    std::atomic<std::uint32_t> active_workers_{0};
};

} // namespace tatara::service
