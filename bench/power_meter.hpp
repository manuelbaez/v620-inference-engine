// GPU power for the benchmarks: the sum of the amdgpu cards' average power (hwmon power1_average, microwatts),
// sampled on a thread, integrated over a timed region. The cache this engine keeps is justified by energy (a
// recompute runs four GPUs at full power for tens of seconds), so a benchmark reports joules next to seconds.
// GPUs only: the host's CPUs and memory are not metered here. The cards' own averaging window is about a second, so
// regions shorter than that read coarsely.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace qw {

class PowerMeter {
public:
    struct Result {
        double seconds = 0, avg_w = 0, kilojoules = 0;
        bool ok = false;
    };

    PowerMeter() {
        for (int i = 0; i < 64; ++i) {
            const std::string dir = "/sys/class/hwmon/hwmon" + std::to_string(i);
            std::ifstream name(dir + "/name");
            std::string n;
            if (name && std::getline(name, n) && n == "amdgpu" && std::ifstream(dir + "/power1_average"))
                files_.push_back(dir + "/power1_average");
        }
    }
    ~PowerMeter() { stop(); }
    bool available() const { return !files_.empty(); }

    void start() {
        stop();
        if (files_.empty()) return;
        running_ = true;
        joules_ = 0;
        t0_ = std::chrono::steady_clock::now();
        sampler_ = std::thread([this] {
            auto last = std::chrono::steady_clock::now();
            while (running_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                const auto now = std::chrono::steady_clock::now();
                joules_ += watts() * std::chrono::duration<double>(now - last).count();
                last = now;
            }
        });
    }

    Result stop() {
        Result r;
        if (!sampler_.joinable()) return r;
        running_ = false;
        sampler_.join();
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        r.kilojoules = joules_ / 1e3;
        r.avg_w = r.seconds > 0 ? joules_ / r.seconds : 0;
        r.ok = true;
        return r;
    }

private:
    double watts() const {
        double w = 0;
        for (const std::string &f : files_) {
            std::ifstream in(f);
            double uw = 0;
            if (in >> uw) w += uw / 1e6;
        }
        return w;
    }

    std::vector<std::string> files_;
    std::thread sampler_;
    std::atomic<bool> running_{false};
    double joules_ = 0;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace qw
