/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include "band.h"
#include "coada.h"
#include <string.h>

/* Aloca un element de tip TCelula2 si returneaza pointerul aferent */
TLista2 AlocCelula2(char x) {
    TLista2 aux = (TLista2) malloc(sizeof(TCelula2));
    aux->pre = NULL;
    aux->urm = NULL;
    if (!aux) {
        return NULL;
    }
    aux->info = x;
    aux->pre = aux->urm = NULL;
    return aux;
}
/* Creeaza santinela pentru lista folosita */
TLista2 InitLista2() {
    TLista2 aux = (TLista2) malloc(sizeof(TCelula2));

    if (!aux) {
        return NULL;
    }
    aux->pre = NULL;
    aux->urm = NULL;
    return aux;
}

void DistrugeLista2(TBanda *aL) {
    TLista2 p = (*aL)->list->urm, aux=NULL;
    while (p != NULL) {         /* distrugere elementele listei */
        aux = p;
        p = p->urm;
        free(aux);
    }

    free((*aL)->list);                  /* distrugere santinela */

    (*aL)->list = NULL;
    (*aL)->deget = NULL;
    free(*aL);
    *aL = NULL;
}
