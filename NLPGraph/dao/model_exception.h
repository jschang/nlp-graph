//
//  Exception.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/17/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef NLPGraph_Exception_h
#define NLPGraph_Exception_h

#include <boost/exception/all.hpp>
#include "../nlpgraph.h"

namespace NLPGraph {
namespace Dao {

typedef struct ModelException : boost::exception, std::exception {
    std::string msg;
    const char *what() const noexcept { return msg.c_str(); };
} ModelExceptionType;

}}
#endif
