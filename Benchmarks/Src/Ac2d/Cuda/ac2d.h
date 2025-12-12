// Ac2D is the acoustic modeling object
//=====================================

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

  float * dp;     // Stress 
  float * dvx;     // x-component of particle velocity
  float * dvy;     // y-component of particle velocity
  float * dexx;    // time derivative of strain x-component
  float * deyy;    // time derivative of strain y-component
  float * dgammax;
  float * dgammay;
  float * dthetax;
  float * dthetay;
  int ts;             // Timestep no
};

// Ac2dNew creates a new Ac2d object
//
  struct ac2d *Ac2dNew(struct model *Model);

// Ac2dSolve computes the pressure at the next timestep
//
  int Ac2dSolve(struct ac2d *Ac2d, struct model *Model,struct src *Src, 
                 struct rec *Rec,int nt, int l, int sresamp, int resamp);
