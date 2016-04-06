const char *kSmithWatermanUtilOpenCLSource = BOOST_COMPUTE_STRINGIZE_SOURCE(

typedef struct smith_waterman {
    uint flags;
    __global char  *logOut;
    uint logLength;
    uint refWidth;
    uint opWidth;
    uint candCount;
    uint uniqueCount;
    ulong costMatrixCoords[2];
    ulong globalOffset;
    __constant ulong *reference;
    __global ulong *candidates;
    __global long *matrices;
    __global long *costMatrix;
    __global ulong *distsAndOps;
    __global ulong *uniques;
} smith_waterman_type;

long costMatrix_indexOf(smith_waterman_type *self, ulong id);
void costMatrix_getCoordsFor(smith_waterman_type *self, size_t offset, ulong *thisCoords);

long costMatrix_indexOf(smith_waterman_type *self, ulong id) {
    ulong searchWidth = self->uniqueCount/2;
    ulong currentPos = (self->uniqueCount/2)-1;
    while(searchWidth>0) {
        if(id < self->uniques[currentPos]) {
            searchWidth /= 2;
            currentPos -= searchWidth;
        }
        if(id > self->uniques[currentPos]) {
            searchWidth /= 2;
            currentPos += searchWidth;
        }
        if(id == self->uniques[currentPos]){
            return currentPos;
        }
        if(searchWidth==1) {
            return -1;
        }
    }
    return -1;
}

void costMatrix_getCoordsFor(smith_waterman_type *self, size_t offset, ulong *thisCoords) {
    // reversed so, processed as xy, then y
    ulong trim = offset, multi = 0;
    int i = 0, j = 0;
    for(i = 1; i>=0; i--) {
        multi = 1;
        for(j=i-1; j>=0; j--) {
            multi *= self->uniqueCount;
        }
        thisCoords[i] = trim / multi;
        trim = trim % multi; 
    } 
}

);