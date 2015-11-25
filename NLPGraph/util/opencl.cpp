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

void OpenCL::log(OpenCLDeviceInfo &thisDeviceInfo) {
    
    LoggerType logger = LoggerType(boost::log::keywords::channel="NLPGraph::Util::OpenCL::deviceInfo");
    
    std::string tmpString = "";
    
    LOG(severity_level::normal) << "Device Id                          : " << thisDeviceInfo.id;
    if(thisDeviceInfo.type & CL_DEVICE_TYPE_CPU) {
        tmpString+="CPU";
    }
    if(thisDeviceInfo.type & CL_DEVICE_TYPE_GPU) {
        tmpString+=std::string(tmpString.length()>0?",":"")+std::string("CPU");
    };
    if(thisDeviceInfo.type & CL_DEVICE_TYPE_ACCELERATOR) {
        tmpString+=std::string(tmpString.length()>0?",":"")+std::string("ACCELERATOR");
    }
    if(thisDeviceInfo.type & CL_DEVICE_TYPE_DEFAULT) {
        tmpString+=std::string(tmpString.length()>0?",":"")+std::string("DEFAULT");
    }
    LOG(severity_level::normal) << "CL_DEVICE_TYPEs                    : " << tmpString;
    LOG(severity_level::normal) << "CL_DEVICE_AVAILABLE                : " << thisDeviceInfo.available;
    LOG(severity_level::normal) << "CL_DEVICE_COMPILER_AVAILABLE       : " << thisDeviceInfo.compilerAvailable;
    LOG(severity_level::normal) << "CL_DEVICE_PROFILE                  : " << (thisDeviceInfo.fullProfile ? "TRUE" : "FALSE");
    LOG(severity_level::normal) << "CL_DEVICE_VERSION >=1.1            : " << (thisDeviceInfo.supportsVer1_1 ? "TRUE" : "FALSE");
    LOG(severity_level::normal) << "CL_DEVICE_EXTENSIONS               : " << (*thisDeviceInfo.extensions);
    LOG(severity_level::normal) << "CL_DEVICE_LOCAL_MEM_SIZE           : " << thisDeviceInfo.localMemSize;
    LOG(severity_level::normal) << "CL_DEVICE_GLOBAL_MEM_SIZE          : " << thisDeviceInfo.globalMemSize;
    LOG(severity_level::normal) << "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE    : " << thisDeviceInfo.globalMemCacheSize;
    LOG(severity_level::normal) << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE : " << thisDeviceInfo.maxConstantBufferSize;
    LOG(severity_level::normal) << "CL_DEVICE_MAX_COMPUTE_UNITS        : " << thisDeviceInfo.computeUnits;
    LOG(severity_level::normal) << "CL_DEVICE_MAX_ITEM_SIZES           : " << NLPGraph::Util::String::str(thisDeviceInfo.maxWorkItemSizes,3);
}

void OpenCL::deviceInfo(cl_device_id id, OpenCLDeviceInfoType &thisDeviceInfo) {

    std::string tmpString;
    
    boost::compute::device thisDevice = boost::compute::device(id);
    
    thisDeviceInfo.id = id;
    
    thisDeviceInfo.type = thisDevice.get_info<cl_device_type>(CL_DEVICE_TYPE);
    
    // verify device availability CL_DEVICE_AVAILABLE
    thisDeviceInfo.available = thisDevice.get_info<cl_bool>(CL_DEVICE_AVAILABLE);
        
    // verify device has a compiler CL_DEVICE_COMPILER_AVAILABLE
    thisDeviceInfo.compilerAvailable = thisDevice.get_info<cl_bool>(CL_DEVICE_COMPILER_AVAILABLE);
        
    // verify device supports full profile CL_DEVICE_PROFILE
    tmpString = thisDevice.get_info<std::string>(CL_DEVICE_PROFILE);
    if(tmpString.compare("FULL_PROFILE")!=0) {
        thisDeviceInfo.fullProfile = false;
    } else {
        thisDeviceInfo.fullProfile = true;
    }
        
    // verify that device supports "OpenCL 1.1" CL_DEVICE_VERSION
    tmpString = thisDevice.get_info<std::string>(CL_DEVICE_VERSION);
    if(tmpString.compare("OpenCL 1.1")!=0 || tmpString.compare("OpenCL 1.2")!=0) {
        thisDeviceInfo.supportsVer1_1 = true;
    } else {
        thisDeviceInfo.supportsVer1_1 = false;
    }
        
    tmpString = thisDevice.get_info<std::string>(CL_DEVICE_EXTENSIONS);
    
    thisDeviceInfo.extensions.reset( new std::string( tmpString.c_str()) );
    thisDeviceInfo.localMemSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);
    thisDeviceInfo.globalMemSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
    thisDeviceInfo.globalMemCacheSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
    thisDeviceInfo.maxConstantBufferSize = thisDevice.get_info<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    thisDeviceInfo.computeUnits = thisDevice.get_info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
    clGetDeviceInfo(thisDeviceInfo.id,CL_DEVICE_MAX_WORK_ITEM_SIZES,(size_t)(sizeof(size_t)*3),(void*)&thisDeviceInfo.maxWorkItemSizes,NULL);
}

bool OpenCL::bestDeviceInfo(OpenCLDeviceInfoType &bestDevice) {

    LoggerType logger = LoggerType(boost::log::keywords::channel="NLPGraph::Util::OpenCL::bestDeviceInfo");

    cl_uint platformCount;
    cl_platform_id* platforms;
    cl_uint deviceCount;
    cl_device_id* devices;
    int i=0, j=0;
    bool ret = false;
    
    // reset our two requirements components
    bestDevice.computeUnits = 0;
    bestDevice.type = 0;
 
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
            OpenCL::deviceInfo(devices[j],thisDeviceInfo);
            thisDeviceInfo.platformId = platforms[i];
            
            if(!thisDeviceInfo.available
                || !thisDeviceInfo.compilerAvailable
                || !thisDeviceInfo.fullProfile
                || !thisDeviceInfo.supportsVer1_1) {
                continue;
            }
            
            if( thisDeviceInfo.type&CL_DEVICE_TYPE_CPU && thisDeviceInfo.computeUnits > bestDevice.computeUnits ) {
                bestDevice = thisDeviceInfo;
                ret = true;
            }
        }
        free(devices);
    }
    free(platforms);
    return ret;
}

cl_context OpenCL::contextWithDeviceInfo(OpenCLDeviceInfoType &deviceInfo) {

    LoggerType logger = LoggerType(boost::log::keywords::channel="NLPGraph::Util::OpenCL::contextWithDeviceInfo");

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

    LoggerType logger = LoggerType(boost::log::keywords::channel="NLPGraph::Util::OpenCL::default_error_handler");

    LOG(severity_level::critical) << errinfo;
    OpenCLException except;
    except.msg = std::string(errinfo);
    throw except;
}

boost::compute::program OpenCL::createAndBuildProgram(std::string src, boost::compute::context ctx) {

    LoggerType logger = LoggerType(boost::log::keywords::channel="NLPGraph::Util::OpenCL::createAndBuildProgram");

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