// Ac2D is the acoustic modeling object
//=====================================

#define NTIMER 10
#define  DXP 1
#define  DXM 2
#define  DYP 3
#define  DYM 4
#define  VX  5
#define  VY  6
#define  STRESS 7

struct ac2d {
  float * p;     // Stress 
  float * vx;     // x-component of particle velocity
  float * vy;     // y-component of particle velocity
  float * exx;    // time derivative of strain x-component
  float * eyy;    // time derivative of strain y-component
  float * gammax;
  float * gammay;
  float * thetax;
  float * thetay;
  int ts;             // Timestep no
  float *timer;
};

// Ac2dNew creates a new Ac2d object
//
  struct ac2d *Ac2dNew(struct model *Model);

// Ac2dPrtime prints kernel timing
  int Ac2dPrtime(struct ac2d *Ac2d);

// Ac2dSolve computes the pressure at the next timestep
//
   int Ac2dSolve(struct ac2d *Ac2d, struct model *Model,struct src *Src, 
                 struct rec *Rec,int nt, int l);
