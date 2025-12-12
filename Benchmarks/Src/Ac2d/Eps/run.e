#
# Run -- eps interface for the run time library
# The run time library is written in C, so this is just
# the eps interface. Most of the routines in the library
# are unix system calls and math functions.
# RunMalloc -- Allocate memory
#char [*] RunMalloc(int nb):end 
 
#
# RunFree -- Deallocate memory
#char [*] RunFree(int nb):end 
 
#
# RunCreate -- create a file
def int RunCreate(char [*] name):
  pass
 
#
# RunClock-- measure elapsed time
def float RunClock():
  pass
 
#
# RunOpen -- open a fileend 
 
def int RunOpen(char [*] name, char [*] mode):
  pass
 
#
# RunClose -- close a file
def int RunClose(int fd):
  pass
 
def int RunRead(int fd, int lbuff, char [*] buffer):
  pass
 
#RunRead reads in  lbuff characters into the
#buffer array from a file with descriptor  fd.
#The return value is the number of characters actually read.
#If an error has occured ERR will be returned.
#
def int RunWrite(int fd, int lbuff, char [*] buffer):
  pass
#
#RunWrite writes lbuff from the buffer array.
#The return value is the number of characters actually written.
# ERR is returned whenever an error has occured.
 
def int RunSeek(int fd, int pos, int flag):
  pass
#
# RunSeek sets the file position to pos bytes
# relative to the start of the file (flag=0), to the current position
# (flag=1) or relative to the end of the file (flag=2).
#  ERR is returned whenever an error has occured.
# Otherwise is the file position returned.
#
def char [*] RunGetenv(char [*] name):
 pass
# RunGetenv returns the value of the environment
# variable contained in the string name.

# RunGetnt gets the number of threads from the
# environment variable NTHREADS.
def int RunGetnt():
  pass

# RunGetnb gets the number of threads from the
# environment variable NTHREADS.
def int RunGetnb():
  pass
 
#
# RunStrcmp -- compare strings
def int RunStrcmp(char [*] s, char [*] t):
  pass
 
#
# RunStrlen -- string length
def int RunStrlen(char [*] s):
  pass

#
# RunExit cleans up and exits.
def int RunExit():
  pass
 
# RunSystem
def int RunSystem(char[*] cmd):
  pass
 
