# Rec object

# Imports
import libe
import model

struct rec :
    int nr;         # No of receivers
    int [*] rx;     # Receiver x-postions
    int [*] ry;     # Receiver y-postions 
    int fd;         # Snapshot output file descriptor
    int nt;         # No of time samples
    float [*,*] p;  # Pressure p[i,j] time sample no j at
                    # position no i 
    float [*,*] wrk; # Work array
    int   resamp;    # Resample factor for receivers
    int   sresamp;   # Resample factor for snapshots
    int pit;         # Next time sample to be recorded

def struct rec RecNew(int [*] rx, int [*] ry, int nt,          \
                  int resamp, int sresamp, char [*] file) :

# RecNew is the constructor for receiver objects.
#
# Arguments:
#   Model:  Model object
#   rx:     Integer array with position of receivers in the 
#           x-direction (gridpoints)
#   ry:     Integer array with position of receivers in the 
#           y-direction (gridpoints)
#   nt:     No of time samples in the receiver data
#   resamp: Resample factor relative to the modelling time sample interval
#   file:   File name for snap shots
#
#  Returns: Receiver object  

  struct rec Rec;

  Rec = new(struct rec);
  Rec.nr = len(rx,0);

  Rec.rx = rx;
  Rec.ry = ry;
  Rec.nt = nt;
  if((Rec.nt > 0) && (Rec.nr > 0)):
    Rec.p = new(float [Rec.nr,Rec.nt]);
  Rec.resamp = resamp;
  Rec.pit = 0;

  if(sresamp > 0):
    Rec.sresamp = sresamp;
    Rec.fd = LibeOpen(file,"w");
  
  return(Rec);

def int RecReceiver(struct rec Rec,int it, float [*,*] p):

# RecReciver records data at the receiver
#
# Arguments: 
#  Rec:    : Receiver object
#  it      : Current time step
#  p:      : Pressure data at time step no it
#
# Returns  : Integer (OK or ERR)

  int pos;
  int ixr,iyr;

  if(Rec.resamp <= 0):
    return(ERR);

  if(Rec.pit > Rec.nt-1):
    return(ERR);

  if(LibeMod(it,Rec.resamp) == 0):
    for (pos=0;pos<Rec.nr; pos=pos+1):  
      ixr=Rec.rx[pos];
      iyr=Rec.ry[pos];
      Rec.p[pos,Rec.pit] = p[ixr,iyr];       
    Rec.pit = Rec.pit+1;
  return(OK);

def int RecSave(struct rec Rec, char [*] file):

# Recsave stores receiver recording on file
#
# Arguments: 
#  Rec:    : Receiver object
#  file    : Output file name
#
# Returns  : Integer (OK or ERR)

  int fd;
  int n;

  if(Rec.resamp <= 0):
    return(ERR);

  fd = LibeOpen(file,"w");
  n = len(Rec.p,0)*len(Rec.p,1);
  LibeWrite(fd,4*n,cast(char [4*n],Rec.p));
  LibeClose(fd);

  return(OK);

def int RecSnap(struct rec Rec,int it, float [*,*] snp):

  # RecSnap records snapshots
  #
  # Arguments: 
  #  Rec:    : Receiver object
  #  it      : Current time step       
  #  snp     : Pressure data
  # Returns  : Integer (OK or ERR)
  int n;
  int Nx, Ny;
  char [*] tmp;
  
  if (Rec.sresamp <= 0):
    return(ERR);

  Nx = len(snp,0);
  Ny = len(snp,1);
  n = Nx*Ny;
  if(LibeMod(it,Rec.sresamp) == 0):
    tmp = cast(char [4*n],snp);
    LibeWrite(Rec.fd,4*n,tmp);
    snp = cast(float [Nx,Ny],tmp)
  return(OK);
