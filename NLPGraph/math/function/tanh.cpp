//
//  tanh.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#include <boost/compute.hpp>
#include "tanh.h"
#include <math.h>

namespace NLPGraph {
namespace Math {
namespace Function {

int TanH::id() {
    return kFunctionTanH;
}
double TanH::operator()(double a) {
    return std::tanh(a);
}
double TanH::derivative(double a) {
    return 1.0 - std::pow(std::tanh(a),2.0);
}
std::string TanH::opencl() {
    return BOOST_COMPUTE_STRINGIZE_SOURCE(
        return tanh(value);
    );
}
std::string TanH::opencl_derivative() {
    return BOOST_COMPUTE_STRINGIZE_SOURCE(
        return 1.0 - pow(tanh(value),2.0);
    );
}

}}}