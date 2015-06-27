//
//  symbol_provider.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__symbol_provider__
#define __NLPGraph__symbol_provider__

#include "../nlpgraph.h"
#include "model.h"

namespace NLPGraph {
namespace Dao {

class SymbolProvider {
public:
    virtual ~SymbolProvider() {}
public:
    virtual NLPGraph::Dto::Symbol* fetchById(uint64_t) { return nullptr; };
};

}}

#endif /* defined(__NLPGraph__symbol_provider__) */
