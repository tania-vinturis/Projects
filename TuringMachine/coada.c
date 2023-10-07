/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include "coada.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

TCoada* InitQ () { 
  TCoada* c = (TCoada*)malloc(sizeof(TCoada));          
  if ( ! c ) return NULL;                  
  c->inc = c->sf = NULL;
  return c;         
}

int IntrQ(TCoada *c, char *x) { 
  TLista aux = NULL;
  aux = (TLista)malloc(sizeof(TCelula));      /* aloca o noua celula */
  aux->info = malloc(sizeof(x));

  if ( ! aux) return 0;             /* alocare imposibila -> "esec" */
  strcpy(aux->info, x);
  aux->urm = NULL;

  if (c->sf != NULL)          /* coada nevida */
    c->sf->urm = aux;                   /* -> leaga celula dupa ultima din coada */
  else                              /* coada vida */
    c->inc = aux;                    /* -> noua celula se afla la inceputul cozii */
  c->sf = aux;  	            /* actualizeaza sfarsitul cozii */
  return 1;                         /* operatie reusita -> "succes" */
}

void DistrQ(TCoada **c) {
  TLista p = NULL, aux = NULL;
  p = (*c)->inc;
  while(p)
  {
    aux = p;
    p = p->urm;
    free(aux);
  }
  free(*c);
  *c = NULL;
}
