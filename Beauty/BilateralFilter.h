#ifndef __BILATERALFILTER__
#define __BILATERALFILTER__

#include "Common.h"

void BilateralFilter(Image &in, Image &out, int r, double sigma_space, double sigma_color, int num = 1);


#endif
