#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <stack>
#include <ANN/ANN.h>

using namespace std;

int main(int argc, char *argv[])
{
    int k = 256, dim = 2, minNs = 6, nPts = 203052;
    
    // READ 
    ANNpointArray pts = annAllocPts(nPts,dim);
    FILE *fp = fopen("tdm_pts.dat","rb");
    for (int i = 0; i < nPts; i++) {
        fread(&pts[i][0],sizeof(double),dim,fp);
    } fclose(fp);

    // KDTREE
    ANNkd_tree  *kdTree = new ANNkd_tree(pts, nPts, dim);
    ANNidxArray   nnIdx = new ANNidx[k];
    ANNdistArray  nnDis = new ANNdist[k];
    ANNdist         rSq = argc > 1 ? atof(argv[1]) : 1.44;
    
    bool discovered[nPts], expanding = false;
    int  groups[nPts], groupID = 0, cur, numNs;
    stack <int> visiting;

    // DBSCAN
    for (int i = 0; i < nPts; i++) {
        if (discovered[i]) continue; 
        discovered[i] = true;
        visiting.push(i);
        while (!visiting.empty()){
            cur = visiting.top(); visiting.pop();        
            numNs = kdTree->annkFRSearch(pts[cur], rSq, k, nnIdx);
            if (numNs < minNs) {
                groups[cur] = -1;
            } else {
                if (!expanding) { groupID++; expanding = true; }
                groups[cur] = groupID;
                for (int j = 0; j < numNs; j++){
                    if (!discovered[nnIdx[j]]) {
                        visiting.push(nnIdx[j]);
                        discovered[nnIdx[j]] = true;
                    }
                }
            }
        } expanding = false;
    }

    // WRITE
    fp = fopen("clusters.dat","wb");
    fwrite(&groups,sizeof(int),nPts,fp);
    fclose(fp);
    
    annClose(); 
    return 0;
}
