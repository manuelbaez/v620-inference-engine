// Adds the seconds a scope took to a counter (the per-request timings of the Session).
#pragma once

#include <chrono>

namespace qw {

class Stopwatch {
public:
    explicit Stopwatch(double *total) : total_(total), t0_(std::chrono::steady_clock::now()) {}
    ~Stopwatch() { *total_ += seconds(); }
    Stopwatch(const Stopwatch &) = delete;
    Stopwatch &operator=(const Stopwatch &) = delete;
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }

private:
    double *total_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace qw
