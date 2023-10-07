/*VÎNTURIȘ Tania-Lorena - 311CB*/
typedef struct celula
{ 
  char *info;
  struct celula* urm;
} TCelula, *TLista;

typedef struct coada
{
  TLista inc, sf;    /* adresa primei si ultimei celule */
} TCoada;

TCoada* InitQ ();
int IntrQ(TCoada *c, char *x);
void DistrQ(TCoada **c);
