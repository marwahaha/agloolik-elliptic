# agloolik-elliptic
Finite Element Method for Arbitrary Second-Order Elliptic PDEs

Author: Adam G. Peddle
Current version: 1.0
Date: 17 March 2015

agloolik-elliptic is a method for computation of abritrary
elliptic PDEs of second order using a 2-D linear triangular
finite element method.

Invocation of the program requires a proper config file
as well as a mesh file, which may be created by the
*simpleMesher.py* file. Unfortunately, the mesher is still
rather user-unfriendly. Calling takes the form:

python3 agloolik controlFileName.ctr

Agloolik runs only with Python3 and is not backwards-compatible.
Agloolik depends on Numpy and Scipy. The meshing similarly
requires the meshpy package, available from:

http://mathema.tician.de/software/meshpy/

At the moment, there are no known bugs and agloolik permits the
use of arbitrary forcing and boundary conditions, which must be
declared in valid numpy-compatible code in the config file.
