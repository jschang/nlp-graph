//
//  tanh.hpp
//  NLPGraph
//
//  Created by Jonathan Schang on 11/26/15.
//
//

#ifndef tanh_hpp
#define tanh_hpp

#include "../function_1d.h"

namespace NLPGraph {
namespace Math {
namespace Function {

class TanH : public NLPGraph::Math::Function1D {
public:
    int id();
    double operator()(double a);
    double derivative(double a);
    std::string opencl();
    std::string opencl_derivative();
};

}}}

#endif /* tanh_hpp */
