#include "Common.h"

//----------------------------------------------------
/**
\brief   sum of box
\param   pSrc                 [in]  input image
\param   pDest                [out] output image
\param   nWidth               [in]  width
\param   nHeight              [in]  height
\param   r                    [in]  radius for sum
\return  void
*/
//-----------------------------------------------------
void BoxSum(double *pSrc, double *pDest, int nWidth, int nHeight, int r)
{
    int nLength = nWidth * nHeight;
    double *pTemp = new double[nLength];

    for(int j = 0; j < nWidth; j++)
    {
        pTemp[j] = pSrc[j];
    }

    for(int i = 1; i < nHeight; i++)
    {
        for(int j = 0; j < nWidth; j++)
        {
            pTemp[i * nWidth + j] = pTemp[(i - 1) * nWidth + j] + pSrc[i * nWidth + j];
        }
    }

    for(int i = 0; i <= r; i++)
    {
        for(int j = 0; j < nWidth; j++)
        {
            pDest[i * nWidth + j] = pTemp[(r + i) * nWidth + j];
        }
    }

    for(int i = r + 1; i < nHeight - 1 - r; i++)
    {
        for(int j = 0; j < nWidth; j++)
        {
            pDest[i * nWidth + j] = pTemp[(r + i) * nWidth + j] - pTemp[(i - r - 1) * nWidth + j];
        }
    }

    for(int i = nHeight - r - 1; i < nHeight; i++)
    {
        for(int j = 0; j < nWidth; j++)
        {
            pDest[i * nWidth + j] = pTemp[(nHeight - 1) * nWidth + j] - pTemp[(i - r - 1) * nWidth + j];
        }
    }

    for(int i = 0; i < nHeight; i++)
    {
        pTemp[i * nWidth] = pDest[i * nWidth];
    }

    for(int j = 1; j < nWidth; j++)
    {
        for(int i = 0; i < nHeight; i++)
        {
            pTemp[i * nWidth + j] = pTemp[i * nWidth + j - 1] + pDest[i * nWidth + j];
        }
    }

    for(int i = 0; i < nHeight; i++)
    {
        for(int j = 0; j <= r; j++)
        {
            pDest[i * nWidth + j] = pTemp[i * nWidth + j + r];
        }
    }

    for(int j = r + 1; j < nWidth - 1 - r; j++)
    {
        for(int i = 0; i < nHeight; i++)
        {
            pDest[i * nWidth + j] = pTemp[i * nWidth + j + r] - pTemp[i * nWidth + j - r - 1];
        }
    }

    for(int j = nWidth - r - 1; j < nWidth; j++)
    {
        for(int i = 0; i < nHeight; i++)
        {
            pDest[i * nWidth + j] = pTemp[i * nWidth + nWidth - 1] - pTemp[i * nWidth + j - r - 1];
        }
    }

    delete []pTemp;
}

//----------------------------------------------------
/**
\brief   GuideFilter
\param   in                   [in]  input image
\param   guide                [in]  output image
\param   out                  [out] guide image
\param   r                    [in]  search radius
\param   eps                  [in]  eps
\return  void
\ref     Guided Image Filtering (ECCV, 2010)
*/
//-----------------------------------------------------
void GuideFilter(Image &in, Image &guide, Image &out, int r, double eps)
{
    int size = in.width * in.height;
    int width = in.width;
    int height = in.height;

    Image temp_in, temp_guide;
    copy(in, temp_in);
    temp_in.data = (unsigned char *)malloc(size * in.channels * sizeof(unsigned char));
    memcpy(temp_in.data, in.data, size * in.channels * sizeof(unsigned char));
    copy(in, temp_guide);
    temp_guide.data = (unsigned char *)malloc(size * in.channels * sizeof(unsigned char));
    memcpy(temp_guide.data, guide.data, size * in.channels * sizeof(unsigned char));
    copy(in, out);

    double *pP = (double *)malloc(size * sizeof(double));
    double *pI = (double *)malloc(size * sizeof(double));
    double *pmeanP = (double *)malloc(size * sizeof(double));
    double *pmeanI = (double *)malloc(size * sizeof(double));
    double *pN  = (double *)malloc(size * sizeof(double));
    double *pIP = (double *)malloc(size * sizeof(double));
    double *pI2 = (double *)malloc(size * sizeof(double));
    double *pA = (double *)malloc(size * sizeof(double));
    double *pB = (double *)malloc(size * sizeof(double));


    for(int t = 0; t < in.channels; t++)
    {
        for(int i = 0; i < size; i++)
        {
            double f1 = temp_in.data[i + t * size] / 255.0;
            double f2 = temp_guide.data[i + t * size] / 255.0;

            pP[i] = f1;
            pI[i] = f2;
            pN[i]  = 1.0;
            pIP[i] = f1 * f2;
            pI2[i] = f2 * f2;
        }

        BoxSum(pP, pmeanP, width, height, r);
        BoxSum(pI, pmeanI, width, height, r);
        BoxSum(pN, pN, width, height, r);
        BoxSum(pIP, pIP, width, height, r);
        BoxSum(pI2, pI2, width, height, r);

        for(int i = 0; i < size; i++)
        {
            pmeanP[i] /= pN[i];
            pmeanI[i] /= pN[i];
            pIP[i] /= pN[i];
            pI2[i] /= pN[i];

            pA[i] = (pIP[i] - pmeanI[i] * pmeanP[i])/(pI2[i] - pmeanI[i] * pmeanI[i] + eps);
            pB[i] = pmeanP[i] - pA[i] * pmeanI[i];
        }

        BoxSum(pA, pA, width, height, r);
        BoxSum(pB, pB, width, height, r);

        for(int i = 0; i < size; i++)
        {
            pA[i] /= pN[i];
            pB[i] /= pN[i];

            int res = 255 * (pA[i] * pI[i] + pB[i]);
            out.data[i + t * size] = CLIP8(res);
        }
    }

    free(temp_in.data);
    free(temp_guide.data);
    free(pP);
    free(pI);
    free(pmeanP);
    free(pmeanI);
    free(pN);
    free(pIP);
    free(pI2);
    free(pA);
    free(pB);
}

//----------------------------------------------------
/**
\brief   GuideFilter(use input image as guide image)
\param   in                   [in]  input image
\param   out                  [out] output image
\param   r                    [in]  search radius
\param   eps                  [in]  eps
\param   num                  [in]  iterative number
\return  void
\ref     Guided Image Filtering (ECCV, 2010)
*/
//-----------------------------------------------------
void GuideFilter(Image &in, Image &out, int r, double eps, int num)
{
    Image temp;
    copy(in, temp);
    temp.data = (unsigned char *)malloc(in.width * in.height * in.channels * sizeof(unsigned char));
    memcpy(temp.data, in.data, in.width * in.height  * in.channels * sizeof(unsigned char));

    while(num--)
    {
        GuideFilter(temp, temp, out, r, eps);
        memcpy(temp.data, out.data, in.width * in.height  * in.channels * sizeof(unsigned char));
    }

    free(temp.data);
}
