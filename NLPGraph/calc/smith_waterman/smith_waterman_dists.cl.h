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
    ulong widthPlusOne = self.refWidth + 1;
    ulong matricesStartIdx = self.globalOffset * (widthPlusOne*widthPlusOne);
    ulong matricesLastIdx = (self.globalOffset+1) * (widthPlusOne*widthPlusOne);
    ulong distanceIdx = (self.opWidth+1) * self.globalOffset;
    ulong opsStartIdx = distanceIdx + 3;
    ulong curCoords[2] = {0,0};
    ulong maxValOffset = 0;
    long maxVal = 0;
    printf("matrix start:%lu end:%lu\n",matricesStartIdx,matricesLastIdx);
    for(ulong i = matricesStartIdx; i<matricesLastIdx; i++) {
        //printf("o:%lu, c:%lu, r:%lu, v:%lu\n",i,i%widthPlusOne,i/widthPlusOne,self.matrices[i]);
        // gte because i want to get as far out into the matrix as possible
        // no pun intended
        if(maxVal<=self.matrices[i]) {
            maxValOffset = i;
            maxVal = self.matrices[i];
        }
    }
    maxValOffset -= matricesStartIdx;
    curCoords[0] = maxValOffset % widthPlusOne;
    curCoords[1] = maxValOffset / widthPlusOne;
    printf("max value: %lu, max value offset:%lu, c:%lu, r:%lu\n",maxVal,maxValOffset,curCoords[0],curCoords[1]);
    ulong dist = 0;
    self.distsAndOps[distanceIdx+1] = curCoords[0];
    self.distsAndOps[distanceIdx+2] = curCoords[1];
    for(int c=curCoords[0]; c>0; ) {
        for(int r=curCoords[1]; r>0; ) {
            ulong curVal = self.matrices[matricesStartIdx+(r*widthPlusOne)+c];
            printf("curCoords:%lu,%lu curVal:%lu ",c,r,curVal);
            long cands[3] = {0,0,0};
            //ulong curOpsIdx = opsStartIdx+(r*c);
            cands[0] = self.matrices[matricesStartIdx+(((r-1)*widthPlusOne)+(c-1))];
            cands[1] = self.matrices[matricesStartIdx+(((r-1)*widthPlusOne)+(c))];
            cands[2] = self.matrices[matricesStartIdx+(((r)*widthPlusOne)+(c-1))];
            ulong idx = util_maxIdxLong(cands,3);
            if(idx==2) {
                // left, implying insertion
                dist++;
                if(c!=0) c--;
            }
            if(idx==1) {
                // upward, implying deletion
                dist++;
                if(r!=0) r--;
            }
            if(idx==0) {
                if(c!=0) c--;
                if(r!=0) r--;
            }
            printf("op:%lu\n",idx);
            self.distsAndOps[opsStartIdx++] = idx+2;
        }
    }
    self.distsAndOps[opsStartIdx++] = 1;
    self.distsAndOps[distanceIdx] = dist;
    printf("total distance %lu",dist);
}

);