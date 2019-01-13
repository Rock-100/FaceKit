#ifndef __GUIDEFILTER__
#define __GUIDEFILTER__

#include "Common.h"

///Sum of Box
void BoxSum(double *pSrc, double *pDest, int nWidth, int nHeight, int r);

///GuideFilter
void GuideFilter(Image &in, Image &guide, Image &out, int r, double eps);
void GuideFilter(Image &in, Image &out, int r, double eps, int num = 1);

#endif

