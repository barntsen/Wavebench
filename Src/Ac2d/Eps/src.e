# Src object

# Imports
import libe

struct src :
  float [*] Src;
  int [*] Sx;
  int [*] Sy;
  int Ns;

def struct src SrcNew(float [*] source, int [*] sx, int [*] sy):

  # SrcNew creates a new source object

  int i;
  struct src Src;
  Src = new(struct src);
  Src.Src = new(float [len(source,0)]);
  for (i=0; i< len(source,0); i=i+1):
    Src.Src[i] = source[i];

  Src.Sx = sx;
  Src.Sy = sy;
  Src.Ns = len(sx,0);
  
  
  return(Src);

def int SrcDel(struct src Src):

  # SrcDel deletes a source object

  delete(Src); 
  return(OK);

def int Srcricker(float [*] source, float t0, float f0, int nt, float dt):

  # Ricker pulse

  float t;
  float w0;
  float arg;
  int i;

  for(i=0; i<nt; i=i+1):
    t = cast(float,i)*dt-t0;
    w0 = 2.0*3.14159*f0;
    arg = w0*t; 
    source[i] = (1.0-0.5*arg*arg)*LibeExp(-0.25*arg*arg);
  return(OK);
