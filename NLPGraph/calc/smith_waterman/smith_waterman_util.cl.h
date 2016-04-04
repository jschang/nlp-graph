const char *kSmithWatermanUtilOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

typedef struct smith_waterman {
    uint flags;
    __global char  *logOut;
    uint logLength;
    uint refWidth;
    uint opWidth;
    uint candCount;
    uint uniqueCount;
    __constant ulong *reference;
    __global ulong *candidates;
    __global ulong *costMatrix;
    __global ulong *distsAndOps;
    __global ulong *uniques;
} smith_waterman_type;

uint costMatrix_indexOf(smith_waterman_type &self, ulong id);
void costMatrix_getCoordsFor(smith_waterman_type &self, size_t offset, uint[] &thisCoords);

uint costMatrix_indexOf(smith_waterman_type &self, ulong id) {
    uint searchWidth = self.uniqueCount/2;
    uint currentPos = self.uniqueCount/2;
    while(searchWidth) {
        if(id < self.uniques[currentPos]) {
            // the id, if it exists, is in the first half of this section
            searchWidth /= 2;
            currentPos -= searchWidth;
        } else
        if(id > self.uniques[currentPos]) {
            // the id, if it exists, is in the second half of this section
            searchWidth /= 2;
            currentPos += searchWidth;
        } else {
            return currentPos;
        }
    }
    return 0;
}

void costMatrix_getCoordsFor(smith_waterman_type &self, size_t offset, uint[] &thisCoords) {
    // reversed so, processed as xy, then y
    ulong trim = offset, multi = 0;
    int i = 0, j = 0;
    for(i = 2-1; i>=0; i--) {
        multi = 1;
        for(j=i-1; j>=0; j--) {
            multi *= self.uniqueCount;
        }
        thisCoords[i] = trim / multi;
        trim = trim % multi; 
    } 
}

);