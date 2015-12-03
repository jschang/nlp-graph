//
//  function_1d.h
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#ifndef function_1d_h
#define function_1d_h

#include "../nlpgraph.h"

namespace NLPGraph {
namespace Math {

typedef enum {
    kFunctionTanH,
    kFunctionLogistic
} FunctionId;

class Function1D {
public:
    static Function1DPtr factory(FunctionId id);
    virtual int id();
    virtual double operator()(double a);
    virtual double inverse(double a);
    virtual double derivative(double a);
    virtual std::string opencl();
    virtual std::string opencl_inverse();
    virtual std::string opencl_derivative();
};

}}

#endif /* function_1d_h */
