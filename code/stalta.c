#include <stdlib.h>

float *stalta_recursiv(float *data, int nsta, int nlta, int N)
{
    float *fc = malloc((sizeof(float) * N));
    float sta, lta;
    sta = 0;
    lta = 0;
    
    int i;
    for (i = 0; i < N; i++)
    {
        sta = (*(data + i) - sta) / nsta + sta;
        lta = (*(data + i) - lta) / nlta + lta;
        if (i > nlta && lta != 0)
            *(fc + i) = sta / lta;
        else
            *(fc + i) = 0;
    }

    return fc;
}

float *stalta_allen(float *data, int nsta, int nlta, int N)
{
    float average(float *data, int N);
    float *fc = malloc((sizeof(float) * N));
    float sta, lta;
    int i;

    for (i = 0; i < nlta; i++)
        *(fc + i) = 0;

    for (i = nlta; i < N; i++)
    {
        sta = average(data + i - nsta, nsta);
        lta = average(data + i - nlta, nlta);
        *(fc + i) = sta / lta;
    }

    return fc;
}

float average(float *data, int n)
{
    float s = 0.0;
    int i;
    for (i = 0; i < n; i++)
        s += *(data + i);
    return s / n;
}

void freeptr(void *ptr)
{
    free(ptr);
}