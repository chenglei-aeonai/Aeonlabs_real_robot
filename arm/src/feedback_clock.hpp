#pragma once
#include <chrono>
#include <cmath>

// Tracks producer samples, independently of how often state is republished.
class FeedbackClock {
 public:
    using Clock = std::chrono::steady_clock;
    void observe(double stamp, Clock::time_point now = Clock::now()) {
        if (!std::isfinite(stamp) || stamp < 0 || stamp < previous_) {
            fault_ = true;
        } else if (stamp > previous_) {
            previous_ = stamp;
            advanced_ = now;
        }
    }
    double age(Clock::time_point now = Clock::now()) const {
        return std::chrono::duration<double>(now - advanced_).count();
    }
    bool healthy(bool connected, Clock::time_point now = Clock::now()) const {
        return connected && !fault_ && previous_ >= 0 && age(now) <= .25;
    }
 private:
    double previous_ = -1;
    bool fault_ = false;
    Clock::time_point advanced_ = Clock::now();
};
