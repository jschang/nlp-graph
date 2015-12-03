//
//  logistic.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#include "logistic.h"

#include <boost/compute.hpp>
#include "tanh.h"
#include <math.h>

namespace NLPGraph {
namespace Math {
namespace Function {

int Logistic::id() {
    return kFunctionLogistic;
}
double Logistic::operator()(double a) {
    return 1.0 / (1.0 + std::pow(M_E,-a));
}
double Logistic::derivative(double a) {
    return (*this)(a) * (1.0 - (*this)(a));
}
std::string Logistic::opencl() {
    return BOOST_COMPUTE_STRINGIZE_SOURCE(
        { return 1.0 / (1.0 + pow(M_E,-value)); }
    );
}
std::string Logistic::opencl_derivative() {
    return BOOST_COMPUTE_STRINGIZE_SOURCE(
        {
            float t = 1.0 / (1.0 + pow(M_E,-value))
            return t * (1.0 - 1);
        }
    );
}

}}}