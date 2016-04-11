const char *kSmithWatermanDetermineDistancesOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

__kernel void calc_smith_waterman_distances(
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
    self.candidates = candidates;
    self.matrices = matrices;
    self.costMatrix = costMatrix;
    self.distsAndOps = distsAndOps;
    self.uniques = uniques;
    self.globalOffset = get_global_id(0);
    
    ulong matricesStartIdx = self.globalOffset * self.refWidth * self.refWidth;
    ulong matricesLastIdx = (self.globalOffset+1) * self.refWidth * self.refWidth - 1;
    ulong distanceIdx = (self.opWidth+1) * self.globalOffset;
    ulong opsStartIdx = (self.opWidth+1) * self.globalOffset + 1;
    ulong curCoords[2] = {0,0};
    ulong maxValOffset = 0;
    long maxVal = 0;
    for(ulong i = matricesStartIdx; i<matricesLastIdx; i++) {
        if(maxVal<self.matrices[i]) {
            maxValOffset = i;
            maxVal = self.matrices[i];
        }
    }
    maxValOffset -= matricesStartIdx;
    util_getCoordsForOffset(&self, maxValOffset, &curCoords[0]);
    for(int c=curCoords[0]; c>0; ) {
        for(int r=curCoords[1]; r>0; ) {
            printf("curCoords: %lu, %lu\n",c,r);
            long cands[3] = {0,0,0};
            ulong curOpsIdx = opsStartIdx+(r*c);
            cands[0] = self.matrices[matricesStartIdx+(((r-1)*self.refWidth)+(c-1))];
            cands[1] = self.matrices[matricesStartIdx+(((r-1)*self.refWidth)+(c))];
            cands[2] = self.matrices[matricesStartIdx+(((r)*self.refWidth)+(c-1))];
            if(cands[1]>cands[0]) {
                // upward, implying deletion
                self.distsAndOps[distanceIdx]++;
                c--;
            }
            if(cands[2]>cands[0]) {
                // left, implying insertion
                self.distsAndOps[distanceIdx]++;
                r--;
            }
            self.distsAndOps[curOpsIdx] = util_maxIdxLong(cands,3);
            if(self.distsAndOps[curOpsIdx]==0) {
                c--;
                r--;
            }
        }
    }
}

);