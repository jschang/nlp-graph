//
//  opencl.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/25/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__opencl__
#define __NLPGraph__opencl__

#include "../nlpgraph.h"
#include <boost/compute.hpp>

namespace NLPGraph {
namespace Util {

typedef struct OpenCLException : boost::exception, std::exception {
    std::string msg;
    const char *what() const noexcept { return msg.c_str(); };
} OpenCLExceptionType;

typedef struct OpenCLDeviceInfo {
    cl_device_id id;
    cl_platform_id platformId;
    cl_ulong globalMemCacheSize; // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
    cl_ulong globalMemSize;      // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong localMemSize;       // CL_DEVICE_LOCAL_MEM_SIZE
    cl_ulong maxConstantBufferSize; // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    cl_bool available;           // CL_DEVICE_AVAILABLE
    cl_bool compilerAvailable;   // CL_DEVICE_COMPILER_AVAILABLE
    cl_uint computeUnits;        // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_bool fullProfile;         // CL_DEVICE_PROFILE
    cl_bool supportsVer1_1;      // CL_DRIVER_VERSION
} OpenCLDeviceInfoType;

class OpenCL {
public:
    static void deviceInfo(cl_device_id id, OpenCLDeviceInfoType &thisDeviceInfo);
    static bool bestDeviceInfo(OpenCLDeviceInfoType &bestDevice);
    static cl_context contextWithDeviceInfo(OpenCLDeviceInfoType &deviceInfo);
    static boost::compute::program createAndBuildProgram(std::string src, boost::compute::context ctx);
    static boost::compute::program getSupportLibrary(boost::compute::context ctx);
public:
    static void default_error_handler (
        const char *errinfo, const void *private_info, 
        size_t cb, void *user_data
    );
};

}}

#endif /* defined(__NLPGraph__opencl__) */
