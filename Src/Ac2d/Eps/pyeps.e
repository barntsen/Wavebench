
import libe                 # Library interface

#Empty main program (Necessary for linking)
def int Main(struct MainArg [*] MainArgs) :
  return(1);


def char [*] PyepsCre1ds(int Nx):

  # PepsCre1ds creates a string buffer
  #
  # Parameter:
  #  Nx      : No of characters in buffer
  #
  # Returns:
  #  str     : String buffer

  char [*] str;
  str = new(char [Nx+1]);
  str[Nx] = cast(char,0);
  return(str);


def int PyepsDel1ds(char [*] arr):

  # PepsDel1ds deletes string buffer
  #
  # Parameter:
  #  arr      : String buffer
  #
  # Returns: integer equal to 1

   delete(arr);
   return(1);



def int PyepsCopy1ds(char [*] arr, char [*] out):

  # PepsCopy1ds makes a copy of a 1d char array
  #
  # Parameter:
  #   arr : Input array
  #   out : Output array
  # Returns: 
  #  integer equal to 1
  int nx;
  int i;

  nx=len(out,0);
  for(i=0; i< nx; i=i+1):
   out[i] = arr[i];
  
  return(1);
     
def int [*] PyepsCre1di(int Nx):

  # PepsCre1di creates integer 1D array
  #
  # Parameter:
  #  int Nx      : No of integers in array
  #
  # Returns:
  #  int [*] arr     : Integer array

  int [*] tmp;
  tmp = new(int [Nx]);
  return(tmp);

def int PyepsDel1di(int [*] arr):

  # PepsDel1di deletes 1D integer array
  #
  # Parameter:
  #  int [*] arr      : Integer array
  #
  # Returns: integer equal to 1
     
   delete(arr);
   return(1);

def int PyepsCopy1di(int [*] arr, int [*] out):
   
  # PepsCopy1di makes a copy of a 1D array
  # Parameter:
  #   arr : Input array
  #   out : Output array
  #
  # Returns: 
  #   integer equal to 1
  int nx;
  int i;

  nx=len(out,0);
  for(i=0; i< nx; i=i+1):
   out[i] = arr[i];
  
  return(1);
     
def float [*] PyepsCre1df(int Nx):

  # PepsCre1di creates float 1D array
  #
  # Parameter:
  #  int Nx      : No of floats in array
  #
  # Returns:
  #  float [*] arr     : Float array
   
  float [*] tmp;
  tmp=new(float [Nx]);
  return(tmp);


# PepsDel1df deletes 1D float array
def int PyepsDel1df(float [*] arr):

  # Parameter:
  #  float [*] arr      : Float array
  #
  # Returns: integer equal to 1
     
   delete(arr);
   return(1);


def int PyepsCopy1df(float [*] arr, float [*] out):

  # PepsCopy1df makes a copy of a 1D float array
  #
  # Parameter:
  #   arr: Input array
  #   out: Output array
  # Returns
  #   integer equal to 1 on success
  int nx;
  int i;

  nx=len(out,0);
  for(i=0; i< nx; i=i+1):
   out[i] = arr[i];
  
  return(1);

def float [*,*] PyepsCre2df(int Nx, int Ny):

  # PepsCre2df creates float 2D array
  #
  # Parameter:
  #  int Nx      : No of floats in array 1st dim
  #  int Ny      : No of floats in array 2nd dim
  #
  # Returns:
  #  float [*,*] arr     : Float array
  #

  return(new(float [Nx,Ny]));


def int PyepsDel2df(float [*,*] arr):

  # PepsDel2df deletes 2D float array
  #
  # Parameter:
  #  float [*,*] arr      : Float array
  #
  # Returns: integer equal to 1
  #  
   delete(arr);
   return(1);


def int PyepsCopy2df(float [*,*] arr, float [*,*] out):

  # PepsCopy2df makes a copy of a 2D float array
  #
  # Parameter:
  #  arr: Input array
  #  out: Output array 
  #
  # Returns: 
  # integer equal to 1 on success
     
  int nx,ny;
  int i,j;

  nx=len(out,0);
  ny=len(out,1);
  for(j=0; j< ny; j=j+1):
    for(i=0; i< nx; i=i+1):
      out[i,j] = arr[i,j];
  
  return(1);


