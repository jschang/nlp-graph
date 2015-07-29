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

// because nvidia is laggard in updating
// probably due to pushing cuda...
// they are apparently the microsoft of
// gpu manufacturers.
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#ifdef __MACH__
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif
// we absolutely need to avoid 1.2, because of NVIDIA's lazy-ass
#ifdef CL_VERSION_1_2
    #undef CL_VERSION_1_2
    #ifndef CL_VERSION_1_1
        #warning "CL_VERSION_1_1 is NOT defined.  Defining it, hoping that boost::compute's ifdefs alone will see us through"
        #define CL_VERSION_1_1 1
    #endif
#endif

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
    cl_ulong globalMemCacheSize;    // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
    cl_ulong globalMemSize;         // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong localMemSize;          // CL_DEVICE_LOCAL_MEM_SIZE
    cl_ulong maxConstantBufferSize; // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    cl_bool available;              // CL_DEVICE_AVAILABLE
    cl_bool compilerAvailable;      // CL_DEVICE_COMPILER_AVAILABLE
    cl_uint computeUnits;           // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_bool fullProfile;            // CL_DEVICE_PROFILE
    cl_bool supportsVer1_1;         // CL_DRIVER_VERSION
    size_t maxWorkItemSizes[3];     // CL_DEVICE_MAX_WORK_ITEM_SIZES
    
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
