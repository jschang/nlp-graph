//
//  opencl.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/25/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#define BOOST_LOG_DYN_LINK
#include "logger.h"
#include "opencl.h"
#include "../util/string.h"
#include <boost/exception/all.hpp>

using namespace NLPGraph::Util;

#define LOG(Sev) BOOST_LOG_SEV(logger,Sev) << __PRETTY_FUNCTION__ << " "

namespace NLPGraph {
namespace Util {

LoggerType logger(boost::log::keywords::channel="NLPGraph::Util::OpenCL");

void OpenCL::deviceInfo(cl_device_id id, OpenCLDeviceInfoType &thisDeviceInfo) {

    std::string tmpString;
    boost::compute::device thisDevice(id);
    memset(&thisDeviceInfo,0,sizeof(OpenCLDeviceInfoType));
    
    thisDeviceInfo.id = id;
        LOG(severity_level::normal) << "device id                          :" << thisDeviceInfo.id;
        
    // verify device availability CL_DEVICE_AVAILABLE
    thisDeviceInfo.available = thisDevice.get_info<cl_bool>(CL_DEVICE_AVAILABLE);
        LOG(severity_level::normal) << "CL_DEVICE_AVAILABLE                :" << thisDeviceInfo.available;
        
    // verify device has a compiler CL_DEVICE_COMPILER_AVAILABLE
    thisDeviceInfo.compilerAvailable = thisDevice.get_info<cl_bool>(CL_DEVICE_COMPILER_AVAILABLE);
        LOG(severity_level::normal) << "CL_DEVICE_COMPILER_AVAILABLE       :" << thisDeviceInfo.compilerAvailable;
        
    // verify device supports full profile CL_DEVICE_PROFILE
    tmpString = thisDevice.get_info<std::string>(CL_DEVICE_PROFILE);
    if(tmpString.compare("FULL_PROFILE")!=0) {
        thisDeviceInfo.fullProfile = false;
    } else {
        thisDeviceInfo.fullProfile = true;
    }
        LOG(severity_level::normal) << "CL_DEVICE_PROFILE                  :" << tmpString;
        
    // verify that device supports "OpenCL 1.1" CL_DEVICE_VERSION
    tmpString = thisDevice.get_info<std::string>(CL_DEVICE_VERSION);
    if(tmpString.compare("OpenCL 1.1")!=0 || tmpString.compare("OpenCL 1.2")!=0) {
        thisDeviceInfo.supportsVer1_1 = true;
    } else {
        thisDeviceInfo.supportsVer1_1 = false;
    }
        LOG(severity_level::normal) << "CL_DEVICE_VERSION                  :" << tmpString;
    thisDeviceInfo.extensions = boost::shared_ptr<std::string>( new std::string( thisDevice.get_info<std::string>(CL_DEVICE_EXTENSIONS) ) );
        LOG(severity_level::normal) << "CL_DEVICE_EXTENSIONS               :" << *thisDeviceInfo.extensions.get();    
    thisDeviceInfo.localMemSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
        LOG(severity_level::normal) << "CL_DEVICE_LOCAL_MEM_SIZE           :" << thisDeviceInfo.localMemSize;
    thisDeviceInfo.globalMemSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
        LOG(severity_level::normal) << "CL_DEVICE_GLOBAL_MEM_SIZE          :" << thisDeviceInfo.globalMemSize;
    thisDeviceInfo.globalMemCacheSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
        LOG(severity_level::normal) << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE    :" << thisDeviceInfo.globalMemCacheSize;
    thisDeviceInfo.maxConstantBufferSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
        LOG(severity_level::normal) << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE :" << thisDeviceInfo.maxConstantBufferSize;
    thisDeviceInfo.computeUnits = thisDevice.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
        LOG(severity_level::normal) << "CL_DEVICE_MAX_COMPUTE_UNITS        :" << thisDeviceInfo.computeUnits;
    memset(&thisDeviceInfo.maxWorkItemSizes,0,sizeof(size_t)*3);
    /*
    clGetDeviceInfo(cl_device_id    // device ,
                cl_device_info  // param_name , 
                size_t          // param_value_size , 
                void *          // param_value ,
                size_t *        // param_value_size_ret ) CL_API_SUFFIX__VERSION_1_0;
    */
    clGetDeviceInfo(thisDeviceInfo.id,CL_DEVICE_MAX_WORK_ITEM_SIZES,(size_t)(sizeof(size_t)*3),(void*)&thisDeviceInfo.maxWorkItemSizes,NULL);
        LOG(severity_level::normal) << "CL_DEVICE_MAX_ITEM_SIZES           :" << NLPGraph::Util::String::str(thisDeviceInfo.maxWorkItemSizes,3);
}

bool OpenCL::bestDeviceInfo(OpenCLDeviceInfoType &bestDevice) {

    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    int i=0, j=0;
    bool ret = false;
    bzero(&bestDevice,sizeof(OpenCLDeviceInfoType));
 
    // get all platforms
    clGetPlatformIDs(0, NULL, &platformCount);
    platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
    clGetPlatformIDs(platformCount, platforms, NULL);
    for (i = 0; i < platformCount; i++) {
    
        // get all devices
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);
        
        for (j = 0; j < deviceCount; j++) {
        
            OpenCLDeviceInfoType thisDeviceInfo;
            bzero(&thisDeviceInfo,sizeof(OpenCLDeviceInfoType));
            OpenCL::deviceInfo(devices[j],thisDeviceInfo);
            thisDeviceInfo.platformId = platforms[i];
            
            if(!thisDeviceInfo.available
                || !thisDeviceInfo.compilerAvailable
                || !thisDeviceInfo.fullProfile
                || !thisDeviceInfo.supportsVer1_1) {
                LOG(severity_level::normal) << "Device " << thisDeviceInfo.id << " is either not available, has no compiler, or doesn't support OpenCL 1.1";
                continue;
            }
            
            if(thisDeviceInfo.computeUnits > bestDevice.computeUnits) {
                memcpy(&bestDevice,&thisDeviceInfo,sizeof(OpenCLDeviceInfoType));
                ret = true;
            }
        }
        free(devices);
    }
    free(platforms);
    return ret;
}

cl_context OpenCL::contextWithDeviceInfo(OpenCLDeviceInfoType &deviceInfo) {
    // create the context
    cl_int errcode_ret;
    LOG(severity_level::critical) << "Creating clCreateContext for "
        << "platformId:" << deviceInfo.platformId << ", deviceId:" << deviceInfo.id;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM,(cl_context_properties)deviceInfo.platformId,0};
    cl_context clContext = clCreateContext(props,1,&deviceInfo.id,&default_error_handler,0,&errcode_ret);
    if(errcode_ret!=CL_SUCCESS) {
        LOG(severity_level::critical) << "clCreateContext failed with code " << errcode_ret;
        OpenCLException except;
        except.msg = "clCreateContext failed.";
        throw except;
    }
    return clContext;
}

void OpenCL::default_error_handler (
    const char *errinfo, 
    const void *private_info, 
    size_t cb, 
    void *user_data
) {
    LOG(severity_level::critical) << errinfo;
    OpenCLException except;
    except.msg = std::string(errinfo);
    throw except;
}

boost::compute::program OpenCL::createAndBuildProgram(std::string src, boost::compute::context ctx) {
    boost::compute::program bProgram = boost::compute::program::create_with_source(src, ctx);
    try {
        bProgram.build("-cl-std=CL1.1 -Werror");
    } catch(...) {
        std::string buildLog = bProgram.get_build_info<std::string>(CL_PROGRAM_BUILD_LOG,ctx.get_device());
        LOG(severity_level::critical) << buildLog;
        OpenCLExceptionType except;
        except.msg = buildLog;
        throw except;
    }
    return bProgram;
}

}}