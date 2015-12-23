/*
Using cufft to do the discrete sine transform and solve the Poisson equation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include "openacc.h"


