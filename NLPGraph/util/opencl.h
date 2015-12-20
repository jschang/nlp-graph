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

class OpenCLDeviceInfo {
public:
    cl_device_id id = 0;
    cl_platform_id platformId = 0;
    cl_device_type type = 0;                  // CL_DEVICE_TYPE
    cl_ulong globalMemCacheSize = 0;          // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
    cl_ulong globalMemSize = 0;               // CL_DEVICE_GLOBAL_MEM_SIZE
    cl_ulong localMemSize = 0;                // CL_DEVICE_LOCAL_MEM_SIZE
    cl_ulong maxConstantBufferSize = 0;       // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
    cl_bool available = 0;                    // CL_DEVICE_AVAILABLE
    cl_bool compilerAvailable = 0;            // CL_DEVICE_COMPILER_AVAILABLE
    cl_uint computeUnits = 0;                 // CL_DEVICE_MAX_COMPUTE_UNITS
    cl_bool fullProfile = 0;                  // CL_DEVICE_PROFILE
    cl_bool supportsVer1_1 = 0;               // CL_DRIVER_VERSION
    size_t maxWorkItemSizes[3] = {0,0,0};     // CL_DEVICE_MAX_WORK_ITEM_SIZES
    boost::shared_ptr<std::string> extensions;
    OpenCLDeviceInfo(){}
    ~OpenCLDeviceInfo(){}
};
typedef OpenCLDeviceInfo OpenCLDeviceInfoType;

class OpenCL {
public:
    static void deviceInfo(cl_device_id id, OpenCLDeviceInfoType &thisDeviceInfo);
    static bool bestDeviceInfo(OpenCLDeviceInfoType &bestDevice);
    static cl_context contextWithDeviceInfo(OpenCLDeviceInfoType &deviceInfo);
    static boost::compute::program createAndBuildProgram(std::string src, boost::compute::context ctx);
    static void log(OpenCLDeviceInfo &thisDeviceInfo);
    template<class T> static void read(cl_command_queue q, size_t count, T *ptr, cl_mem buf);
    template<class T> static void write(cl_command_queue q, size_t count, T *ptr, cl_mem buf);
    template<class T> static void alloc(cl_context c, size_t count, T **ptr, cl_mem *buf, int clFlags);
public:
    static void default_error_handler (
        const char *errinfo, const void *private_info, 
        size_t cb, void *user_data
    );
};

template<class T>
void OpenCL::read(cl_command_queue q, size_t count, T *ptr, cl_mem buf) {

    cl_int errcode = 0;
    
    errcode = clEnqueueReadBuffer(q, buf, true, 0, sizeof(T)*count, ptr, 0, NULL, NULL);
    if(errcode!=CL_SUCCESS) {
        OpenCLExceptionType except;
        except.msg = "Unable to read buffer";
        throw except;
    }
};

template<class T>
void OpenCL::write(cl_command_queue q, size_t count, T *ptr, cl_mem buf) {

    cl_int errcode = 0;
    
    errcode = clEnqueueWriteBuffer(q, buf, true, 0, sizeof(T)*count, ptr, 0, NULL, NULL);
    if(errcode!=CL_SUCCESS) {
        OpenCLExceptionType except;
        except.msg = "Unable to write buffer";
        throw except;
    }
};

template<class T>
void OpenCL::alloc(cl_context ctx, size_t count, T **ptr, cl_mem *buf, int clFlags) {

    cl_int errcode = 0;
    
    if(ptr && !*ptr) {
        *ptr = new T[count];
        if(!*ptr) {
            OpenCLExceptionType except;
            except.msg = "unable to allocate host ptr";
            throw except;
        }
        memset(*ptr, 0, count * sizeof(T));
    }
    
    if(buf && !*buf) {
        *buf = clCreateBuffer(ctx,clFlags,sizeof(T)*count,*ptr,&errcode);
        if(errcode!=CL_SUCCESS) {
            OpenCLExceptionType except;
            except.msg = "unable to allocate cl_mem object";
            throw except;
        }
    }
};

}}

#endif /* defined(__NLPGraph__opencl__) */
