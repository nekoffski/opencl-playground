#include <kc/core/Log.h>
#include <kc/core/Profiler.h>

void run() {
    PROFILE_FUNCTION();
    LOG_INFO("Hello world!");
}

int main() {
    kc::core::initLogging("opencl-playground");
    kc::core::Profiler profiler;

    run();

    profiler.saveResults("./");
    return 0;
}