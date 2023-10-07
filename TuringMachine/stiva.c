/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "stiva.h"

int Push(TStiva *vf, TCelula2* x){
    TStiva aux = (TStiva)malloc(sizeof(TCelulaStiva));
    if(!aux) return 0;
    aux->info = x;
    aux->urm = NULL;
    if (*vf == NULL){ // tratăm cazul în care stiva este goala
        *vf = aux;
    } else{
        aux->urm = *vf;
        *vf = aux;
    }
    return 1;
}

TCelula2* Pop(TStiva *vf){
    if (*vf == NULL){
        return NULL;
    }
    TCelula2* aux = NULL;
    aux = (*vf)->info;
    TStiva tmp = *vf;
    *vf = (*vf)->urm;
    free(tmp);
    return aux;
}

void DistrugereS(TStiva *s) {
    if (*s == NULL) {
        return;
    }
    TStiva aux;
    while (*s != NULL) {
        aux = *s;
        *s = (*s)->urm;
        free(aux);
    }
    *s = NULL;
}
