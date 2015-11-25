//
//  kohonen_som_data.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "kohonen_som_data.h"

namespace NLPGraph {
    namespace Calc {
        
KohonenSOMData::KohonenSOMData(const boost::compute::context &context, 
        const std::vector<double> &nodeWeights, // product(mapDimensions) * nodeWidth
        const std::vector<uint32_t> &mapDimensions, 
        const int nodeWidth) {
        
    this->_mapDimensions = boost::shared_ptr< std::vector<uint32_t> >( new std::vector<uint32_t>(mapDimensions) );
    this->_nodeWidth = nodeWidth;
    this->_nodeCount = nodeWeights.size()/nodeWidth;
    cl_int err = 0;
    // because i can't cound on the device having cl_khr_fp64
    std::vector<cl_float> floatNodeWeights(nodeWeights.begin(), nodeWeights.end());
    this->_clNodeWeights = clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,(size_t)floatNodeWeights.size()*sizeof(float),(void*)floatNodeWeights.data(),&err);
    if (err!=CL_SUCCESS) {
        Util::OpenCLExceptionType except;
        except.msg = except.msg + "unable to clCreateBuffer _clNodeWeights; ";
        throw except;
    }
    this->_clMapDimensions = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)mapDimensions.size()*sizeof(uint32_t),(void*)mapDimensions.data(),&err);
    if (err!=CL_SUCCESS) {
        Util::OpenCLExceptionType except;
        except.msg = except.msg + "unable to clCreateBuffer _clMapDimensions; ";
        throw except;
    }
}
KohonenSOMData::~KohonenSOMData() {
    if(_clNodeWeights!=0) {
        clReleaseMemObject(_clNodeWeights);
    }
    if(_clNodeWeights!=0) {
        clReleaseMemObject(_clMapDimensions);
    }
}
void KohonenSOMData::fromClMem(const boost::compute::command_queue &commandQueue, std::vector<double> &weights) {
    size_t wc = _nodeCount*_nodeWidth;
    float *result = new float[wc];
    try {
        // find the minimum distance in clOutputData
        cl_int err = clEnqueueReadBuffer(commandQueue, _clNodeWeights, true, 0, wc*sizeof(float), result, 0, NULL, NULL);
        if(err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = except.msg + "Unable to read logBuf; ";
            throw except;
        }
        weights.clear();
        weights.resize(wc);
        std::copy(result,result+wc,weights.begin());
    } catch(...) {
        free(result);
        throw;
    }
    free(result);
}
const int KohonenSOMData::nodeWidth() {
    return _nodeWidth;
}
const uint64_t KohonenSOMData::nodeCount() {
    return _nodeCount;
}
const std::vector<uint32_t>* KohonenSOMData::mapDimensions() {
    return _mapDimensions.get();
}
const cl_mem KohonenSOMData::clNodeWeights() {
    return _clNodeWeights;
}
const cl_mem KohonenSOMData::clMapDimensions() {
    return _clMapDimensions;
}
    
}}