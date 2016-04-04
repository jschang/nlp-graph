function binarySearch(uniques, id) {
    var searchWidth = Math.round(uniques.length / 2);
    var currentPos = Math.round(uniques.length / 2)-1;
    while(searchWidth>0) {
        console.log('id:'+id+',currentId:'+uniques[currentPos]+',searchWidth:'+searchWidth+',currentPos:'+currentPos);
        if(id < uniques[currentPos]) {
            // the id, if it exists, is in the first half of this section
            searchWidth = Math.round(searchWidth / 2);
            currentPos -= searchWidth;
        }
        if(id > uniques[currentPos]) {
            // the id, if it exists, is in the second half of this section
            searchWidth = Math.round(searchWidth / 2);
            currentPos += searchWidth;
        } 
        if(id == uniques[currentPos]) {
            return currentPos;
        }
        if(searchWidth==1 || typeof(uniques[currentPos])=='undefined') {
            return -1;
        }
    }
    return -1;
}

function testThese(uniques) {
    console.log(uniques);
    var k = 0;
    for(var i of uniques) {
        console.log('testing for '+i+', at pos '+k);
        console.log(uniques[binarySearch(uniques,i)]+' == '+uniques[k]);
        k++;
    }
    console.log('testing for 8');
    console.log(uniques[binarySearch(uniques,8)]+' == 8');
    console.log('testing for -1');
    console.log(uniques[binarySearch(uniques,-1)]+' == -1');
    console.log('testing for 100000');
    console.log(uniques[binarySearch(uniques,-1)]+' == 100000');
}
testThese([1,4,7,10,34]);
testThese([1,4,7,10,35, 46,69,102,345,400, 1023,1024]);
