/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include "band.h"

typedef struct celst{
    TCelula2 *info;
    struct celst *urm;
}TCelulaStiva, *TStiva;

int Push(TStiva *vf, TCelula2* x);
TCelula2* Pop(TStiva *vf);
void DistrugereS(TStiva *s);
