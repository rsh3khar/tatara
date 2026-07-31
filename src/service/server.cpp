#include "tatara/service/server.h"

#include "tatara/service/http_message.h"
#include "tatara/service/router.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

namespace tatara::service {
namespace {

// Poll wakes at this cadence so a drain request is noticed even when no
// connection is active.
constexpr int kPollIntervalMilliseconds = 50;
constexpr std::size_t kReadChunkBytes = 16 * 1024;

struct Connection {
    int descriptor{-1};
    std::string inbound;
    bool close_after_response{false};
};

bool write_all(int descriptor, std::string_view payload) {
    std::size_t written = 0;
    while (written < payload.size()) {
        const auto count =
            ::send(descriptor, payload.data() + written, payload.size() - written, MSG_NOSIGNAL);
        if (count <= 0) {
            if (count < 0 && errno == EINTR) {
                continue;
            }
            return false;
        }
        written += static_cast<std::size_t>(count);
    }
    return true;
}

// A client that has gone away shows up as a readable socket that yields either
// zero bytes or an error. Checking it costs one non-blocking peek.
bool peer_gone(int descriptor) {
    char probe = 0;
    const auto count = ::recv(descriptor, &probe, 1, MSG_PEEK | MSG_DONTWAIT);
    if (count == 0) {
        return true;
    }
    if (count < 0) {
        return errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR;
    }
    return false;
}

} // namespace

Server::Server(Configuration configuration, ServerHooks hooks)
    : configuration_(std::move(configuration)), hooks_(std::move(hooks)) {}

Server::~Server() {
    if (listener_ >= 0) {
        ::close(listener_);
    }
}

std::uint16_t Server::port() const {
    return port_;
}

void Server::stop() {
    draining_.store(true);
}

bool Server::start() {
    listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listener_ < 0) {
        return false;
    }
    int reuse = 1;
    ::setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(configuration_.service.port);
    if (::inet_pton(AF_INET, configuration_.service.bind.c_str(), &address.sin_addr) != 1) {
        ::close(listener_);
        listener_ = -1;
        return false;
    }
    if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
        ::close(listener_);
        listener_ = -1;
        return false;
    }
    if (::listen(listener_, static_cast<int>(configuration_.service.queue_depth) + 1) != 0) {
        ::close(listener_);
        listener_ = -1;
        return false;
    }
    sockaddr_in bound{};
    socklen_t length = sizeof(bound);
    if (::getsockname(listener_, reinterpret_cast<sockaddr*>(&bound), &length) == 0) {
        port_ = ntohs(bound.sin_port);
    }
    return true;
}

void Server::run() {
    std::vector<Connection> connections;
    const std::uint64_t connection_limit =
        std::uint64_t{configuration_.service.max_concurrent_requests} +
        configuration_.service.queue_depth;

    while (true) {
        // Draining stops new work but lets in-flight connections finish, which
        // is the difference between a graceful drain and a hangup.
        const bool draining = draining_.load();
        if (draining && connections.empty()) {
            break;
        }

        std::vector<pollfd> descriptors;
        descriptors.reserve(connections.size() + 1);
        if (!draining && connections.size() < connection_limit) {
            descriptors.push_back({listener_, POLLIN, 0});
        }
        for (const auto& connection : connections) {
            descriptors.push_back({connection.descriptor, POLLIN, 0});
        }
        if (descriptors.empty()) {
            break;
        }

        const int ready = ::poll(descriptors.data(), static_cast<nfds_t>(descriptors.size()),
                                 kPollIntervalMilliseconds);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }

        std::size_t index = 0;
        if (!draining && connections.size() < connection_limit) {
            if ((descriptors[0].revents & POLLIN) != 0) {
                const int accepted = ::accept(listener_, nullptr, nullptr);
                if (accepted >= 0) {
                    connections.push_back({accepted, {}, false});
                }
            }
            index = 1;
        }

        std::vector<Connection> survivors;
        survivors.reserve(connections.size());
        for (std::size_t position = 0; position < connections.size(); ++position, ++index) {
            Connection connection = std::move(connections[position]);
            const short events = index < descriptors.size() ? descriptors[index].revents : 0;
            bool keep = true;

            if ((events & (POLLHUP | POLLERR)) != 0) {
                keep = false;
            } else if ((events & POLLIN) != 0) {
                std::string chunk(kReadChunkBytes, '\0');
                const auto count = ::recv(connection.descriptor, chunk.data(), chunk.size(), 0);
                if (count <= 0) {
                    keep = false;
                } else {
                    connection.inbound.append(chunk.data(), static_cast<std::size_t>(count));
                    auto parsed = parse_request(connection.inbound);
                    if (parsed.state == ParseState::Complete) {
                        connection.inbound.erase(0, parsed.consumed_bytes);
                        keep = handle(connection.descriptor, parsed.request);
                    } else if (parsed.state != ParseState::Incomplete) {
                        const bool body_too_large = parsed.state == ParseState::BodyTooLarge;
                        const auto body = render_error(
                            body_too_large ? "request body exceeds the service limit"
                                           : "malformed or unsupported request framing",
                            "invalid_request_error", parse_state_code(parsed.state));
                        write_all(connection.descriptor,
                                  render_response(parse_state_http_status(parsed.state),
                                                  "application/json", body, false));
                        keep = false;
                    }
                }
            }

            if (keep) {
                survivors.push_back(std::move(connection));
            } else {
                ::close(connection.descriptor);
            }
        }
        connections = std::move(survivors);
    }

    for (auto& connection : connections) {
        ::close(connection.descriptor);
    }
    if (listener_ >= 0) {
        ::close(listener_);
        listener_ = -1;
    }
}

bool Server::handle(int descriptor, const HttpRequest& request) {
    const ServiceSnapshot snapshot = hooks_.snapshot ? hooks_.snapshot() : ServiceSnapshot{};
    const Route route = resolve_route(request.method, request.target);

    if (route != Route::ChatCompletions && route != Route::Completions) {
        const auto response = route_operational(route, snapshot);
        const bool keep = request.keep_alive && response.status < 500;
        write_all(descriptor,
                  render_response(response.status, response.content_type, response.body, keep));
        return keep;
    }

    if (!is_ready(snapshot)) {
        const auto refusal = reject(Rejection::NotReady);
        write_all(descriptor,
                  render_response(refusal.status, refusal.content_type, refusal.body, false));
        return false;
    }
    if (!hooks_.completions) {
        const auto body =
            render_error("no completion handler is installed", "server_error", "tatara.internal");
        write_all(descriptor, render_response(500, "application/json", body, false));
        return false;
    }

    Generation generation;
    bool headers_sent = false;
    generation.emit = [&](std::string_view chunk) {
        if (!generation.stream) {
            return true;
        }
        if (!headers_sent) {
            if (!write_all(descriptor, render_sse_headers(false))) {
                return false;
            }
            headers_sent = true;
        }
        return write_all(descriptor, render_sse_event(chunk));
    };
    generation.cancelled = [descriptor]() { return peer_gone(descriptor); };

    const bool produced = hooks_.completions(route, request.body, generation);

    if (generation.stream) {
        if (!headers_sent) {
            write_all(descriptor, render_sse_headers(false));
        }
        write_all(descriptor, render_sse_done());
        return false;
    }
    if (!produced) {
        if (generation.response_status != 200) {
            const bool keep =
                request.keep_alive && generation.response_status < 500;
            write_all(
                descriptor,
                render_response(generation.response_status,
                                generation.response_content_type,
                                generation.body, keep));
            return keep;
        }
        const auto body =
            render_error("the request could not be completed", "server_error", "tatara.internal");
        write_all(descriptor, render_response(500, "application/json", body, false));
        return false;
    }
    const bool keep = request.keep_alive;
    write_all(descriptor,
              render_response(generation.response_status,
                              generation.response_content_type,
                              generation.body, keep));
    return keep;
}

} // namespace tatara::service
