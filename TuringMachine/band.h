/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include <stdio.h>
#include <stdlib.h>

/* definire lista dublu inlantuita cu santinela */
typedef struct celula2{
  char info;
  struct celula2 *pre, *urm;
}TCelula2, *TLista2;

typedef struct banda {
  TLista2 list;
  TCelula2* deget;
} *TBanda;

TLista2 AlocCelula2(char x);
TLista2 InitLista2();
//void DistrugeLista2(TLista2 *aL);
void DistrugeLista2(TBanda *aL);
