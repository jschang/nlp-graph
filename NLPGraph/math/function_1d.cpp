//
//  function_1d.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#include "function_1d.h"
#include "function/tanh.h"
#include "function/logistic.h"

using namespace NLPGraph::Math::Function;

namespace NLPGraph {
namespace Math {

Function1DPtr Function1D::factory(FunctionId id) {
    switch(id) {
    case kFunctionTanH:
        return Function1DPtr(new TanH());
    case kFunctionLogistic:
        return Function1DPtr(new Logistic());
    default:
        NLPGraphExceptionType exc;
        exc.msg = "Unimplemented";
        throw exc; 
    }
}

int Function1D::id() {
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc;
}
double Function1D::operator()(double a) { 
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc; 
}
double Function1D::inverse(double a) { 
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc; 
}
double Function1D::derivative(double a) { 
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc; 
}
std::string Function1D::opencl() {
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc;
}
std::string Function1D::opencl_inverse() {
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc;
}
std::string Function1D::opencl_derivative() {
    NLPGraphExceptionType exc;
    exc.msg = "Unimplemented";
    throw exc;
}

}};