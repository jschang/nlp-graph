//
//  logistic.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#ifndef logistic_hpp
#define logistic_hpp

#include "../function_1d.h"

namespace NLPGraph {
namespace Math {
namespace Function {

class Logistic : public NLPGraph::Math::Function1D {
public:
    int id();
    double operator()(double a);
    double derivative(double a);
    std::string opencl();
    std::string opencl_derivative();
};

}}}

#endif /* logistic_hpp */
