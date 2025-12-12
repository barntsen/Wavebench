import libe    
import model
import src
import rec
import ac2d

def int Main(struct MainArg [*] MainArgs):
  float [*] wavelet;  # Source pulse
  float [*,*] vp;     # 
  float [*,*] rho;    # 
  float [*,*] Q;      # 
  float f0;

  struct model Model; # Model
  struct rec   Rec;   # Receiver
  struct src   Src;   # Source
  int [*] sx, sy;     # Source x,y-coordinates
  int [*] rx, ry;     # Receiver x,y-coordinates
  struct ac2d  Ac2d;  # FD solver
  int Nx,Ny;          # Model dimension in x- and y-directions.
  int nt,ntr;          # No of time steps
  int resamp, sresamp; # Resampling factors for data and snapshot
  float dt, dx;        # Time sampling and space sampling intervals
  int fd;              # File desriptor for source pulse              
  int i;               # Timestep no
  char  [*] tmp;       # Temporary workspace
  float W0;
  int Nb,Rheol;
  int Nr;
  int l;
  float si;
  float t0,tios,tioe,tmods,tmode,tss,tse;         # Timers
  int nbl,nth;

  # Initialize library
  LibeInit();

  t0 = LibeClock();
  # Main modeling parameters
  Nx=LibeAtoi(MainArgs[1].arg); # x-dimensiom
  Ny=LibeAtoi(MainArgs[2].arg); # x-dimensiom

  dx=5.0; # grid interval
  dt=0.0005; # Time sampling
  nt=LibeAtoi(MainArgs[3].arg); # No of timesteps
  l=8;      # Operator length
  f0=25.0;   # Peak frequency
  W0=f0*3.14159*2.0; # Central angular frequency
  Nb = 35;             # Border for PML attenuation
  Rheol = MAXWELL;

  nth=1024
  nbl = 1024
  LibeSetnt(nth);
  LibeSetnb(nbl);

  tios=LibeClock()
  # Read the velocity model
  fd=LibeOpen("vp.bin","r");
  tmp = new(char [4*Nx*Ny]);
  LibeRead(fd,4*Nx*Ny,tmp); 
  vp=cast(float [Nx,Ny], tmp);
  LibeClose(fd);

  # Read the density model
  fd=LibeOpen("rho.bin","r");
  tmp = new(char [4*Nx*Ny]);
  LibeRead(fd,4*Nx*Ny,tmp); 
  rho=cast(float [Nx,Ny], tmp);
  LibeClose(fd);

  # Read the attenuation model
  fd=LibeOpen("q.bin","r");
  tmp = new(char [4*Nx*Ny]);
  LibeRead(fd,4*Nx*Ny,tmp); 
  Q=cast(float [Nx,Ny], tmp);
  LibeClose(fd);
  tioe=LibeClock()

  # Read a source signature from file
  fd=LibeOpen("src.bin","r");
  tmp = new(char [4*nt]);
  LibeRead(fd,4*nt,tmp); 
  wavelet=cast(float [nt], tmp);
  LibeClose(fd);

  # Create a source 
  sx=new(int[1]);
  sy=new(int[1]);
  sx[0]= Nx/2;
  sy[0]= Ny/2;
  Src=SrcNew(wavelet,sx,sy);

  # Create a model  
  tmods=LibeClock()
  Model = ModelNew(vp,rho,Q,dx,dt,W0,Nb,Rheol); 
  si=ModelStability(Model);
  LibePuts(stderr,"Stability index: ");
  LibePutf(stderr,si,"g");
  LibePuts(stderr,"\n");
  LibeFlush(stderr);
  tmode=LibeClock()

  # Create a receiver
  Nr=201;
  rx=new(int[Nr]);
  ry=new(int[Nr]);
  for(i=0; i<Nr; i=i+1):
    rx[i] = i;
    ry[i] = 50;
  resamp= LibeAtoi(MainArgs[4].arg)
  sresamp=LibeAtoi(MainArgs[5].arg)
  if(resamp > 0):
    ntr = nt/resamp; #No of output samples per rec
  else :
    ntr=0

  Rec= RecNew(rx,ry,ntr,resamp,sresamp,"snp.bin");

  # Create solver  
  Ac2d = Ac2dNew(Model);

  tss=LibeClock();
  # Run solver 
  Ac2dSolve(Ac2d, Model, Src, Rec, nt,l);
  tse=LibeClock()

  # Save recording
  if(Rec.resamp > 0) :
    RecSave(Rec,"p.bin");

  LibePuti(stdout,Nx);
  LibePuts(stdout,"\n");

  LibePuti(stdout,Ny);
  LibePuts(stdout,"\n");

  LibePuti(stdout,nt);
  LibePuts(stdout,"\n");

  LibePutf(stdout,tioe-tios,"g"); 
  LibePuts(stdout,"\n");
  LibePutf(stdout,tmode-tmods,"g"); 
  LibePuts(stdout,"\n");
  LibePutf(stdout,tse-tss,"g"); 
  LibePuts(stdout,"\n");
  LibePutf(stdout,LibeClock()-t0,"g");
  LibePuts(stdout,"\n");

  #Ac2dPrtime(Ac2d)

  return(OK);
