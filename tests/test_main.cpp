#include <iostream>
#include <vector>
#include <functional>
#include <string>
#include <cmath>
#include <chrono>

struct TestResult {
    std::string name;
    bool passed;
    std::string message;
    double durationMs;
};

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }

    void addTest(const std::string& name, std::function<void()> test) {
        tests_.push_back({name, test});
    }

    int runAll() {
        std::cout << "==============================\n";
        std::cout << "  FEM Sparse Solver Unit Tests\n";
        std::cout << "==============================\n\n";

        std::vector<TestResult> results;
        int passed = 0;
        int failed = 0;

        for (const auto& test : tests_) {
            TestResult result{test.first, true, "", 0.0};

            auto start = std::chrono::high_resolution_clock::now();
            try {
                test.second();
            } catch (const std::exception& e) {
                result.passed = false;
                result.message = e.what();
            } catch (...) {
                result.passed = false;
                result.message = "Unknown exception";
            }
            auto end = std::chrono::high_resolution_clock::now();
            result.durationMs = std::chrono::duration<double, std::milli>(end - start).count();

            results.push_back(result);

            if (result.passed) {
                passed++;
                std::cout << "[PASS] " << result.name
                          << " (" << result.durationMs << " ms)\n";
            } else {
                failed++;
                std::cout << "[FAIL] " << result.name << "\n";
                std::cout << "       Error: " << result.message << "\n";
            }
        }

        std::cout << "\n==============================\n";
        std::cout << "  Results: " << passed << " passed, "
                  << failed << " failed\n";
        std::cout << "==============================\n";

        return failed > 0 ? 1 : 0;
    }

private:
    struct TestCase {
        std::string name;
        std::function<void()> func;
    };

    std::vector<TestCase> tests_;
};

#define TEST_CASE(name) \
    void test_##name(); \
    namespace { \
        bool registered_##name = []() { \
            TestRunner::instance().addTest(#name, test_##name); \
            return true; \
        }(); \
    } \
    void test_##name()

#define ASSERT_TRUE(cond) \
    do { if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); } while(0)

#define ASSERT_FALSE(cond) \
    do { if (cond) throw std::runtime_error("Assertion failed: not " #cond); } while(0)

#define ASSERT_EQ(a, b) \
    do { if (!((a) == (b))) throw std::runtime_error("Assertion failed: " #a " == " #b); } while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a) - (b)) > (tol)) { \
        std::string msg = "Assertion failed: " #a " ~= " #b " (diff=" + std::to_string(std::abs((a)-(b))) + ")"; \
        throw std::runtime_error(msg); \
    }} while(0)

#define ASSERT_GT(a, b) \
    do { if (!((a) > (b))) throw std::runtime_error("Assertion failed: " #a " > " #b); } while(0)

#define ASSERT_LT(a, b) \
    do { if (!((a) < (b))) throw std::runtime_error("Assertion failed: " #a " < " #b); } while(0)

int main() {
    return TestRunner::instance().runAll();
}
