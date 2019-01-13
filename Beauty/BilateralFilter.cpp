#include "Common.h"

//----------------------------------------------------
/**
\brief   BilateralFilter
\param   in                 [in]  input image
\param   out                [out] output image
\param   r                  [in]  search radius
\param   sigma_space        [in]  variance of space
\param   sigma_color        [in]  variance of color
\param   num                [in]  iterative number
\return  void
\ref     Bilateral Filtering for Gray and Color Images (ICCV, 1998)
*/
//-----------------------------------------------------
void BilateralFilter(Image &in, Image &out, int r, double sigma_space, double sigma_color, int num)
{
    copy(in, out);

    Image temp;
    temp.data = (unsigned char *)malloc(in.height * in.width * in.channels * sizeof(unsigned char));
    copy(in, temp);

    memcpy(temp.data, in.data, in.height * in.width * in.channels * sizeof(unsigned char));

    //temporary data
    int id             = 0;
    int nPatchPixs     = (2 * r + 1) * (2 * r + 1);
    int CurVal         = 0;		                     //current pixel value
    int nCenter        = 0;

    double fSumWeight  = 0;
    double fSumValue   = 0;
    double var         = 0;		                     //variance


    //calculate the distance weight
    double *fdWeight = (double *)malloc(nPatchPixs * sizeof(double));
    for(int x = -r; x <= r; ++x)
    {
        for(int y = -r; y <= r; ++y)
        {
            fdWeight[id] = sigma_space / (sigma_space + x * x + y * y);
            ++id;
        }
    }

    double fcWeight[256];
    for(int i = 0; i < 256; i++)
    {
        fcWeight[i] = sigma_color / (sigma_color + i * i);
    }

    while(num--)
    {
        for(int t = 0; t != temp.channels; ++t)
        {
            int offset = t * temp.height * temp.width;
            for(int i = 0; i != temp.height; ++i)
            {
                for(int j = 0; j != temp.width; ++j)
                {
                    fSumValue = 0;
                    fSumWeight = 0;

                    nCenter = temp.data[i * temp.width + j + offset];

                    for(int x = -r; x <= r; ++x)
                    {
                        for(int y = -r; y <= r; ++y)
                        {
                            int m = CLAMP(i + x, 0, temp.height - 1);
                            int n = CLAMP(j + y, 0, temp.width - 1);
                            id = (x + r) * (2 * r + 1) + y + r;

                            CurVal = temp.data[m * temp.width + n + offset];
                            int nDiff = abs(CurVal - nCenter);
                            double fWeight = fcWeight[nDiff] * fdWeight[id];
                            fSumValue += CurVal * fWeight;
                            fSumWeight += fWeight;
                        }
                    }

                    fSumValue /= fSumWeight;
                    out.data[i * temp.width + j + offset] = CLIP8(int(fSumValue));
                }
            }
        }

        memcpy(temp.data, out.data, in.height * in.width * in.channels * sizeof(unsigned char));
    }

    free(fdWeight);
    free(temp.data);
}
