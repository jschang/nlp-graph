//
//  kohonen_som_result.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/24/15.
//
//

#include "kohonen_som_result.h"
#include "kohonen_som_sample_data.h"
#include "kohonen_som_result.h"

namespace NLPGraph {
    namespace Calc {

KohonenSOMResult::KohonenSOMResult(const boost::compute::context &context, const KohonenSOMSampleDataPtr &data) {
    _indexes = boost::shared_ptr< std::vector<std::vector<uint32_t>> >( new std::vector<std::vector<uint32_t>>(data->count()) );
    _distances = boost::shared_ptr< std::vector<float> >( new std::vector<float>(data->count()) );
}
KohonenSOMResult::~KohonenSOMResult() {
    this->freeClMem();
}
void KohonenSOMResult::freeClMem() {
    if(_clDistances!=0) {
        clReleaseMemObject(_clDistances);
        _clDistances = 0;
    }
    if(_clIndexes!=0) {
        clReleaseMemObject(_clIndexes);
        _clIndexes = 0;
    }
}
void KohonenSOMResult::toClMem(const boost::compute::context &context) {
    cl_int err = 0;
    if(_clDistances==0) {
        this->_clDistances = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,(size_t)_distances->size()*sizeof(float),(void*)_distances->data(),&err);
        if (err!=CL_SUCCESS) {
            Util::OpenCLExceptionType except;
            except.msg = "unable to clCreateBuffer _clDistances; ";
            throw except;
        }
    }
    if(_clIndexes==0) {
        uint32_t dimCount = (*_indexes.get())[0].size();
        uint32_t *indexes = new uint32_t[_indexes->size()*dimCount];
        try {
            for(int i = 0; i<_indexes->size(); i++) {
                for(int j = 0; j<dimCount; j++) {
                    indexes[(i*dimCount)+j] = (*_indexes.get())[i][j];
                }
            }
            this->_clIndexes = clCreateBuffer(
                        context,
                        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                        (size_t)_indexes->size()*dimCount*sizeof(uint32_t),(void*)indexes,&err);
            if (err!=CL_SUCCESS) {
                Util::OpenCLExceptionType except;
                except.msg = "unable to clCreateBuffer _clIndexes; ";
                throw except;
            }
            free(indexes);
        } catch(...) {
            free(indexes);
            throw;
        }
    }
}
std::vector<std::vector<uint32_t>>* KohonenSOMResult::indexes() {
    return _indexes.get();
}
std::vector<float>* KohonenSOMResult::distances() {
    return _distances.get();
}
cl_mem KohonenSOMResult::clDistances() {
    return _clDistances;
}
cl_mem KohonenSOMResult::clIndexes() {
    return _clIndexes;
}

}}