#include <kc/core/Log.h>
#include <kc/core/Profiler.h>
#include <kc/parallel/All.h>

void showDeviceInfo(const cl::Device& device) {
    LOG_INFO("Vendor: {}, Vendor ID: {}, Version: {}", device.getInfo<CL_DEVICE_VENDOR>(),
             device.getInfo<CL_DEVICE_VENDOR_ID>(), device.getInfo<CL_DEVICE_VERSION>());
}

int main() {
    kc::core::initLogging("opencl-playground");
    kc::core::Profiler profiler;
    kc::core::FileSystem fs;

    kc::parallel::Context context;

    context.init();

    constexpr int length = 1000;

    std::vector<float> a, b, c;

    a.resize(length, 5);
    b.resize(length, 3);
    c.resize(length, 0);

    {
        auto dryRun = [&]() {
            {
                PROFILE_REGION(dryRun);
                for (int i = 0; i < length; ++i) c[i] = a[i] + b[i];
            }
            for (auto& cc : c) ASSERT(cc == 8, "invalid value for dry run");
        };

        auto parallelRun = [&]() {
            {
                PROFILE_REGION(parallelRunFull);

                try {
                    static const std::string kernelFile = "../src/kernels/kernel.cl";

                    cl::CommandQueue queue(context.get());

                    kc::parallel::Buffer<float> dA(length, kc::parallel::BufferType::readOnly,
                                                   context);

                    dA.fill(3).bind();

                    kc::parallel::Buffer<float> dB(length, kc::parallel::BufferType::readOnly,
                                                   context);
                    dB.fill(5).bind();

                    kc::parallel::Buffer<float> dC(length, kc::parallel::BufferType::writeOnly,
                                                   context);
                    dC.bind();

                    auto kernel =
                        kc::parallel::Kernel<cl::Buffer, cl::Buffer, cl::Buffer>::fromFile(
                            context, kernelFile, "vadd");

                    {
                        PROFILE_REGION(parallelRunKernelOnly);

                        kernel.get()(cl::EnqueueArgs(queue, cl::NDRange(length)), dA.get(),
                                     dB.get(), dC.get());
                    }

                    dC.readValuesFromGpu(queue);

                    for (auto& cc : dC)
                        ASSERT(cc == 8, "invalid value for parallel run: " + std::to_string(cc));

                } catch (std::exception& e) {
                    LOG_FATAL("{}", e.what());
                }
            }
        };

        dryRun();
        c = std::vector<float>(length, 0);
        parallelRun();
    }

    profiler.saveResults("./");
    return 0;
}