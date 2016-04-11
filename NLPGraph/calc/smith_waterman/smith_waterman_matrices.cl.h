const char *kSmithWatermanCreateMatricesOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

long matrices_similarityFunction(
    smith_waterman_type *self,
    ulong referenceId, ulong candidateId);
    
long matrices_getSimilaryScore(
    smith_waterman_type *self,
    ulong startMatrixOffset,
    ulong col, ulong row,
    ulong colId, ulong rowId);
    
void matrices_getMaxSimilarityScore(
    smith_waterman_type *self,
    ulong startMatrixOffset,
    ulong col, ulong row,
    long *maxes);

__kernel void calc_smith_waterman_matrices(
        uint flags,
        __global char  *logOut,
        uint logLength,
        uint refWidth,
        uint opWidth,
        uint candCount,
        uint uniqueCount,
        __constant ulong *reference,
        __global ulong *candidates,
        __global long *matrices,
        __global long *costMatrix,
        __global ulong *distsAndOps,
        __global ulong *uniques
) { 
    smith_waterman_type self;
    memset(&self,0,sizeof(smith_waterman_type));
    self.flags = flags;
    self.logOut = logOut;
    self.logLength = logLength;
    self.refWidth = refWidth;
    self.opWidth = opWidth;
    self.candCount = candCount;
    self.uniqueCount = uniqueCount;
    self.reference = reference;
    self.candidates = candidates;
    self.matrices = matrices;
    self.costMatrix = costMatrix;
    self.distsAndOps = distsAndOps;
    self.uniques = uniques;
    self.globalOffset = get_global_id(0);
    
    ulong startIdx = self.globalOffset * self.refWidth * self.refWidth;
    ulong candIdx = self.globalOffset * self.refWidth;
    
    for(ulong col=0; col<self.refWidth; col++) {
        ulong candId = self.candidates[candIdx+col];
        
        for(ulong row=0; row<self.refWidth; row++) {
        
            ulong refId = self.reference[row];
            long maxSimilarity[4] = {
                0,0, // max similarity score of prev in row and prev in col
                0,   // current similarity score
                0    // zero
            };
            
            matrices_getMaxSimilarityScore(&self,startIdx,col,row,maxSimilarity);
            maxSimilarity[2] = matrices_getSimilaryScore(
                    &self,
                    startIdx,
                    col, row,
                    refId, candId
                );
            
            long maxScore = -65355;
            for(int i=0; i<4; i++) {
                maxScore = maxScore < maxSimilarity[i] ? maxSimilarity[i] : maxScore;
            }
            
            self.matrices[startIdx+(row*self.refWidth)+col] = maxScore;
        }
    }
}

long matrices_similarityFunction(smith_waterman_type *self, ulong referenceId, ulong candidateId) {
    if(referenceId==candidateId) {
        return +2;
    }
    return -1;
}

long matrices_getSimilaryScore(
    smith_waterman_type *self,
    ulong startMatrixOffset,
    ulong col, ulong row,
    ulong colId, ulong rowId // ref and cand respectively
) {
    if(col==0 || row==0) {
        return 0;
    }
    if(col>0 && row>0) {
        return self->matrices[startMatrixOffset+((row-1)*self->refWidth)+(col-1)]
            + matrices_similarityFunction(self,colId,rowId);
    }
    return 0;
}

void matrices_getMaxSimilarityScore(
    smith_waterman_type *self,
    ulong startMatrixOffset,
    ulong col, ulong row,
    long *maxes
) {
    maxes[0] = -65355; // col
    for(ulong c = 0; c<col; c++) {
        long cur = self->matrices[startMatrixOffset+(row*self->refWidth)+c];
        maxes[0] = cur > maxes[0] ? cur : maxes[0];
    }
    maxes[0] += -1; // W - gap-scoring scheme
    
    maxes[1] = -65355; // row
    for(ulong r = 0; r<row; r++) {
        long cur = self->matrices[startMatrixOffset+(r*self->refWidth)+col];
        maxes[1] = cur > maxes[1] ? cur : maxes[1];
    }
    maxes[1] += -1; // W - gap-scoring scheme
}

);
