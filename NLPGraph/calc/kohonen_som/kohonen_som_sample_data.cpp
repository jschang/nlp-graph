//
//  kohonen_som_sample_data.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "kohonen_som_sample_data.h"

namespace NLPGraph {
    namespace Calc {

KohonenSOMSampleData::KohonenSOMSampleData(const boost::compute::context &context, 
        const std::vector<double> &sampleData, 
        const uint32_t sampleWidth) {
        
    this->_width = sampleWidth;
    this->_count = sampleData.size()/sampleWidth;
    cl_int err = 0;
    // because i can't cound on the device having cl_khr_fp64
    std::vector<float> floatSampleData(sampleData.begin(), sampleData.end());
    this->_clData = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)floatSampleData.size()*sizeof(float),(void*)floatSampleData.data(),&err);
    if (err!=CL_SUCCESS) {
        Util::OpenCLExceptionType except;
        except.msg = "unable to clCreateBuffer _clData; ";
        throw except;
    }
}
KohonenSOMSampleData::~KohonenSOMSampleData() {
    if(this->_clData!=0) {
        clReleaseMemObject(this->_clData);
    }
}
cl_mem KohonenSOMSampleData::clData() { return this->_clData; }
uint KohonenSOMSampleData::width() { return this->_width; }
uint KohonenSOMSampleData::count() { return this->_count; }

}}