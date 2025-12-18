// Differentiator exports

//diff object
struct diff {
  int  l;        // Differentiator length
  int lmax;      // Max differentiator length
  float *coeffs; 
  float *w;     // Differentiator weights 
};
//Methods:
struct diff *DiffNew(int l);
int DiffDxminus(struct diff *Diff, float *A, float *dA, float dx, int nx, int ny);
int DiffDyminus(struct diff *Diff, float *A, float *dA, float dx, int nx, int ny);
int DiffDxplus(struct diff *Diff, float *A, float *dA, float dx,  int nx, int ny);
int DiffDyplus(struct diff *Diff, float *A, float *dA, float dx,  int nx, int ny);
