//
//  symbol_provider_postgres.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__symbol_provider_postgres__
#define __NLPGraph__symbol_provider_postgres__

#include "../../nlpgraph.h"

namespace NLPGraph {
namespace Dao {

class SymbolProviderPostgres : public NLPGraph::Dao::SymbolProvider {
private:
    ModelPtr m_model;
public:
    SymbolProviderPostgres(ModelPtr modelPool);
    virtual ~SymbolProviderPostgres() {}
public: // from SymbolProvider
    Dto::Symbol* fetchById(uint64_t) { return nullptr; }
};

}}

#endif /* defined(__NLPGraph__symbol_provider_postgres__) */
