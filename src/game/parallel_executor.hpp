#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace gunpowder {

// A small persistent worker group for short, index-addressed simulation jobs.
// The calling thread participates in every batch, avoiding per-tick thread
// creation while leaving one hardware thread available to the renderer.
class ParallelExecutor {
public:
    explicit ParallelExecutor(unsigned int maximumWorkers = 8) {
        const unsigned int hardwareThreads =
            std::max(1U, std::thread::hardware_concurrency());
        const unsigned int backgroundWorkers =
            hardwareThreads > 1 ? hardwareThreads - 1 : 0;
        const unsigned int workerTotal =
            std::min(maximumWorkers, backgroundWorkers);
        workers_.reserve(workerTotal);
        for (unsigned int worker = 0;
             worker < workerTotal; ++worker) {
            workers_.emplace_back([this] {
                workerLoop();
            });
        }
    }

    ParallelExecutor(const ParallelExecutor&) = delete;
    ParallelExecutor& operator=(const ParallelExecutor&) = delete;

    ~ParallelExecutor() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        workAvailable_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    [[nodiscard]] unsigned int workerCount() const {
        return static_cast<unsigned int>(workers_.size() + 1);
    }

    template <typename Function>
    void run(std::size_t count, Function&& function) {
        if (count == 0) {
            return;
        }
        if (workers_.empty() || count == 1) {
            for (std::size_t index = 0; index < count; ++index) {
                function(index);
            }
            return;
        }

        {
            std::lock_guard lock(mutex_);
            function_ = std::forward<Function>(function);
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            completedWorkers_ = 0;
            exception_ = nullptr;
            ++generation_;
        }
        workAvailable_.notify_all();

        executeAvailableWork();

        std::unique_lock lock(mutex_);
        workComplete_.wait(lock, [&] {
            return completedWorkers_ == workers_.size();
        });
        function_ = {};
        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

private:
    void executeAvailableWork() noexcept {
        try {
            while (true) {
                const std::size_t index =
                    next_.fetch_add(1, std::memory_order_relaxed);
                if (index >= count_) {
                    break;
                }
                function_(index);
            }
        } catch (...) {
            std::lock_guard lock(exceptionMutex_);
            if (!exception_) {
                exception_ = std::current_exception();
            }
            next_.store(count_, std::memory_order_relaxed);
        }
    }

    void workerLoop() {
        std::size_t observedGeneration = 0;
        while (true) {
            {
                std::unique_lock lock(mutex_);
                workAvailable_.wait(lock, [&] {
                    return stopping_ ||
                           generation_ != observedGeneration;
                });
                if (stopping_) {
                    return;
                }
                observedGeneration = generation_;
            }

            executeAvailableWork();

            {
                std::lock_guard lock(mutex_);
                ++completedWorkers_;
                if (completedWorkers_ == workers_.size()) {
                    workComplete_.notify_one();
                }
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::mutex exceptionMutex_;
    std::condition_variable workAvailable_;
    std::condition_variable workComplete_;
    std::function<void(std::size_t)> function_;
    std::atomic<std::size_t> next_{0};
    std::size_t count_ = 0;
    std::size_t generation_ = 0;
    std::size_t completedWorkers_ = 0;
    std::exception_ptr exception_;
    bool stopping_ = false;
};

} // namespace gunpowder
