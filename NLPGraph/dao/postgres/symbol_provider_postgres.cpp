//
//  symbol_provider_postgres.cpp
//  NLPGraph
//
//  Created by Jonathan Schang on 6/16/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#include "../../nlpgraph.h"

using namespace NLPGraph::Util;

namespace NLPGraph {
namespace Dao {

SymbolProviderPostgres::SymbolProviderPostgres(ModelPtr model) {
    m_model.reset(model.get());
}

}}