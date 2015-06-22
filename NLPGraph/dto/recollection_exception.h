//
//  recollection_exception.h
//  NLPGraph
//
//  Created by Jonathan Schang on 6/19/15.
//  Copyright (c) 2015 Jonathan Schang. All rights reserved.
//

#ifndef NLPGraph_recollection_exception_h
#define NLPGraph_recollection_exception_h

#include "../nlpgraph.h"

namespace NLPGraph {
namespace Dto {

typedef enum {
    RecExceptOpNone,
    RecExceptOpReplace,
    RecExceptOpDelete,
    RecExceptOpInsert,
    RecExceptOpTranspose
} RecExceptOpsEnum;

class RecollectionException {
private:
    uint64_t m_id;
    RecExceptOpsEnum m_operationId;
    RecollectionPtr m_recollection;
    SymbolPtr m_symbol;
    int m_memberIndex;
public:
    RecollectionException(uint64_t id, RecollectionPtr recollection, RecExceptOpsEnum operationId, int memberIndex, SymbolPtr symbol) 
            : m_id(id), m_operationId(RecExceptOpNone), m_recollection(nullptr), m_memberIndex(-1), m_symbol(nullptr) {
        m_symbol = symbol;
        RecollectionException(recollection,operationId,memberIndex);
    }
    RecollectionException(RecollectionPtr recollection, RecExceptOpsEnum operationId, int memberIndex) {
        m_recollection = recollection;
        m_operationId = operationId;
        m_memberIndex = memberIndex;
    }
    uint64_t getId() { return m_id; }
    RecollectionPtr getRecollection() { return m_recollection; }
    SymbolPtr getSymbol() { return m_symbol; }
};

}}

#endif
