// Minimal persistent thread pool with a blocking parallel_for.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace qw {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads = 0) {
        if (n_threads <= 0) n_threads = int(std::thread::hardware_concurrency());
        if (n_threads <= 0) n_threads = 1;
        n_ = n_threads;
        for (int i = 1; i < n_; ++i) workers_.emplace_back([this] { worker(); });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto &t : workers_) t.join();
    }
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    int size() const { return n_; }

    // Calls fn(begin, end) over [0, n) in chunks of at least `grain` items and
    // returns when all of them have run. Not reentrant.
    void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)> &fn) {
        if (n <= 0) return;
        if (grain < 1) grain = 1;
        int64_t chunks = (n + grain - 1) / grain;
        if (chunks > int64_t(n_) * 4) chunks = int64_t(n_) * 4;
        if (chunks <= 1 || n_ == 1) {
            fn(0, n);
            return;
        }
        Job job{&fn, n, chunks};
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = &job;
            ++gen_;
        }
        cv_.notify_all();
        run(job);
        std::unique_lock<std::mutex> lk(mu_);
        job_ = nullptr;  // no new worker can pick it up from here on
        done_cv_.wait(lk, [&] { return job.done == job.chunks && job.active == 0; });
    }

private:
    struct Job {
        const std::function<void(int64_t, int64_t)> *fn;
        int64_t n, chunks;
        std::atomic<int64_t> next{0};
        int64_t done = 0;  // guarded by mu_
        int active = 0;    // workers holding a pointer; guarded by mu_
        Job(const std::function<void(int64_t, int64_t)> *f, int64_t n_, int64_t c) : fn(f), n(n_), chunks(c) {}
    };

    void run(Job &job) {
        int64_t mine = 0;
        for (;;) {
            int64_t c = job.next.fetch_add(1);
            if (c >= job.chunks) break;
            (*job.fn)(job.n *c / job.chunks, job.n * (c + 1) / job.chunks);
            ++mine;
        }
        if (mine) {
            std::lock_guard<std::mutex> lk(mu_);
            job.done += mine;
        }
    }

    void worker() {
        uint64_t seen = 0;
        for (;;) {
            Job *job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                if (stop_) return;
                job = job_;
                if (!job) continue;
                ++job->active;
            }
            run(*job);
            {
                std::lock_guard<std::mutex> lk(mu_);
                --job->active;
            }
            done_cv_.notify_all();
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    Job *job_ = nullptr;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace qw
