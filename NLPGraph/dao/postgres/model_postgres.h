//
//  model_postgres.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/17/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef __NLPGraph__model_postgres__
#define __NLPGraph__model_postgres__

#include "../../nlpgraph.h"
#include "../../util/resource_pool.h"
#include "../model.h"
#include <pqxx/pqxx>

namespace NLPGraph {
namespace Dao {

class ModelPostgres : NLPGraph::Dao::Model {
private:
    Util::ResourcePoolPtr<pqxx::connection*> m_dbPool;
    std::string m_schema;
public:
    ModelPostgres(Util::ResourcePoolPtr<pqxx::connection*> pool, std::string schema);
    ~ModelPostgres();
public:
    bool isCreated();
    bool create();
    bool destroy();
    bool reset();
    
    void prepare();
    
    Dto::InputChannelPtr newInputChannel();
    
    Dto::SymbolPtr newSymbol(Dto::InputChannelPtr channel);
    void addSymbolMember(Dto::SymbolPtr parentSymbol, Dto::SymbolPtr memberSymbol);
    long getSymbolMemberCount(Dto::SymbolPtr symbol);
    
    Dto::RecollectionPtr newRecollection(Dto::SymbolPtr symbol);
    void addRecollectionException(Dto::RecollectionPtr recollection, Dto::RecExceptOpsEnum operationId, int symbolIdx, Dto::SymbolPtr symbol);
};

}}

#endif /* defined(__NLPGraph__model_postgres__) */
