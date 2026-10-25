#include "feedback_clock.hpp"
#include <limits>
#include <stdexcept>

void check(bool condition) {
    if (!condition) throw std::runtime_error("Feedback clock assertion failed");
}
int main() {
    const auto start = FeedbackClock::Clock::now();
    FeedbackClock clock;
    check(!clock.healthy(true, start));
    clock.observe(12., start);
    check(clock.healthy(true, start));
    const auto later = start + std::chrono::milliseconds(300);
    clock.observe(12., later); // Repeated publication cannot refresh a stale sample.
    check(!clock.healthy(true, later));
    clock.observe(12.1, later);
    check(clock.healthy(true, later));
    check(!clock.healthy(false, later));
    clock.observe(11.9, later);
    check(!clock.healthy(true, later));
    clock.observe(12.2, later); // A clock reset remains latched until restart.
    check(!clock.healthy(true, later));
    FeedbackClock invalid;
    invalid.observe(std::numeric_limits<double>::quiet_NaN(), start);
    check(!invalid.healthy(true, start));
}
