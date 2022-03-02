#include <kc/core/Log.h>
#include <kc/core/Profiler.h>

// #define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_TARGET_OPENCL_VERSION 210

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl2.hpp>

void showDeviceInfo(const cl::Device& device) {
    LOG_INFO("Vendor: {}, Vendor ID: {}, Version: {}", device.getInfo<CL_DEVICE_VENDOR>(),
             device.getInfo<CL_DEVICE_VENDOR_ID>(), device.getInfo<CL_DEVICE_VERSION>());
}

int main() {
    kc::core::initLogging("opencl-playground");
    kc::core::Profiler profiler;
    kc::core::FileSystem fs;

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    ASSERT(devices.size() > 0, "Could not find any device that support opencl");

    for (auto& device : devices) showDeviceInfo(device);

    auto& device = devices.front();

    constexpr int length = 1000 * 1000;

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
                    auto kernelSource = fs.readFile(kernelFile);

                    cl::Context context(device);

                    cl::Program program{context, kernelSource, true};
                    cl::CommandQueue queue(context);
                    cl::Buffer dA(context, a.begin(), a.end(), true);
                    cl::Buffer dB(context, b.begin(), b.end(), true);
                    cl::Buffer dC(context, CL_MEM_WRITE_ONLY, sizeof(float) * length);

                    {
                        PROFILE_REGION(parallelRunKernelOnly);

                        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> kernel(program,
                                                                                     "vadd");

                        kernel(cl::EnqueueArgs(queue, cl::NDRange(length)), dA, dB, dC);
                        cl::copy(queue, dC, c.begin(), c.end());
                    }

                } catch (std::exception& e) {
                    LOG_FATAL("{}", e.what());
                }
                for (auto& cc : c) ASSERT(cc == 8, "invalid value for parallel run");
            }
        };

        dryRun();
        c = std::vector<float>(length, 0);
        parallelRun();
    }

    profiler.saveResults("./");
    return 0;
}