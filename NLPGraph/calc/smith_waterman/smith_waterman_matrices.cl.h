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
    
    ulong startIdx = self.globalOffset * (self.refWidth+1) * (self.refWidth+1);
    ulong candIdx = self.globalOffset * self.refWidth;
    
    ulong widthPlusOne = (self.refWidth+1);
    
    for(ulong row=0; row<widthPlusOne; row++) {
    
        // leading col and row are always zero
        if(!row) continue;
        
        // get the current row
        ulong candId = self.candidates[candIdx+(row-1)];
        
        for(ulong col=0; col<widthPlusOne; col++) {
        
            // leading col and row are always zero
            if(!col) continue;
            
            // get the current column candidate id
            ulong refId = self.reference[col-1];
            long maxSimilarity[4] = {
                0, // max for col prev in row (c,0->r)
                0, // max for row prev in col (0->c,r)
                0, // current similarity score
                0  // zero
            };
                        
            matrices_getMaxSimilarityScore(&self,startIdx,col,row,maxSimilarity);
            maxSimilarity[2] = matrices_getSimilaryScore(
                    &self,
                    startIdx,
                    col,   row,
                    refId, candId
                );
            
            long maxScore = -65355;
            for(int i=0; i<4; i++) {
                maxScore = maxScore < maxSimilarity[i] 
                            ? maxSimilarity[i] 
                            : maxScore;
            }
            
            printf("c:%lu,r:%lu ref:%lu, cand:%lu [%li,%li,%li,%li]\n",col,row,refId,candId,maxSimilarity[0],maxSimilarity[1],maxSimilarity[2],maxSimilarity[3]);
            self.matrices[startIdx+(row*widthPlusOne)+col] = maxScore;
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
    // add the final score for the last two
    ulong prev = self->matrices[startMatrixOffset+((row-1)*(self->refWidth+1))+(col-1)];
    // to the similarity function for these two
    long sim = matrices_similarityFunction(self,colId,rowId);
    printf("c:%lu,r:%lu prev:%lu, sim:%li\n",col,row,prev,sim);
    return prev + sim;
}

void matrices_getMaxSimilarityScore(
    smith_waterman_type *self,
    ulong startMatrixOffset,
    ulong col, ulong row,
    long *maxes
) {
    // get max previous in the row up to the current column
    /*
    maxes[0] = -65355; // col
    for(ulong c = 0; c<col; c++) {
        long cur = self->matrices[startMatrixOffset+(row*(self->refWidth+1))+c];
        maxes[0] = cur > maxes[0] ? cur : maxes[0];
    }
    maxes[0] += -2; // W - gap-scoring scheme
    */
    maxes[0] = self->matrices[startMatrixOffset+(row*(self->refWidth+1))+(col-1)] - 2;
    
    // get max previous in the column up to the current row
    /*
    maxes[1] = -65355; // row
    for(ulong r = 0; r<row; r++) {
        long cur = self->matrices[startMatrixOffset+(r*(self->refWidth+1))+col];
        maxes[1] = cur > maxes[1] ? cur : maxes[1];
    }
    maxes[1] += -2; // W - gap-scoring scheme
    */
    maxes[1] = self->matrices[startMatrixOffset+((row-1)*(self->refWidth+1))+(col)] - 2;
}

);
