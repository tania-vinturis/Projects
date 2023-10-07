/*VÎNTURIȘ Tania-Lorena - 311CB*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include "coada.h"
#include "stiva.h"

#define WRITECHAR 6
#define MOVERIGHTCHAR 16
#define MOVELEFTCHAR 15
#define INSERTRIGHT 13
#define INSERTLEFT 12
#define DIMINPUT 500

int MOVE_LEFT(TBanda Banda, TStiva* undoStack) {
    if(Banda->deget->pre->pre == NULL){
        return 0;
    }
    else{
        Push(undoStack, Banda->deget); //adaug in stiva de undo pozitia curenta
        Banda->deget = Banda->deget->pre; //mut degetul la stanga
    }
 }
 
void MOVE_RIGHT(TBanda Banda, TStiva* undoStack) {
    if(Banda->deget->urm == NULL){
        TLista2 aux = AlocCelula2('#'); //daca degetul este pe ultima pozitie, adaug # pe pozitia urmatoare si mut degetul
        aux->urm = NULL;
        aux->pre = Banda->deget; 
        Banda->deget->urm = aux; 
        
    }
    Push(undoStack, Banda->deget); //adaug in stiva de undo pozitia curenta
    Banda->deget = Banda->deget->urm; //mut degetul la dreapta
   
}

void MOVE_LEFT_CHAR(TBanda Banda, char c, FILE* out){
    TLista2 aux = Banda->deget;
    while(aux->pre != NULL && aux->info != c){ //inainteaza la stanga pana gaseste caracterul c dat ca parametru
        aux = aux->pre;
    }
    if(aux->pre == NULL){
        fprintf(out, "ERROR\n");
        return;
    }
    Banda->deget = aux; //mut degetul pe pozitia gasita
}

void MOVE_RIGHT_CHAR(TBanda Banda, char c){
    while(Banda->deget->urm !=NULL && Banda->deget->info !=c){ //inainteaza la dreapta pana gaseste caracterul c
        Banda->deget = Banda->deget->urm; 
    }
    if(Banda->deget->urm == NULL && Banda->deget->info !=c){ //daca "iese din dimensiunea initiala" a benzii, 
	//adauga # pe pozitia uramtoare si muta degetul
        TLista2 aux = AlocCelula2(c);
        aux->info = '#';
        aux->urm = NULL;
        aux->pre = Banda->deget;
        Banda->deget->urm = aux;
        Banda->deget = Banda->deget->urm;
        
    }
}

void WRITE(TBanda Banda, char c, TStiva* undoStack){
    Banda->deget->info = c; //scrie caracterul c pe pozitia curenta
    //verific daca stiva de undo este goala, daca nu e goala o golesc
    if(undoStack!=NULL){
         Pop(undoStack);
    }
}

void INSERT_LEFT(TBanda Banda, char c, FILE *out){
    if(Banda->deget->pre->pre == NULL){ //daca degetul este pe prima pozitie, nu se poate insera la stanga
        fprintf(out, "ERROR\n");
        return;
    }
    else{  
       TLista2 aux = AlocCelula2(c);
       Banda->deget->pre->urm = aux; 
       aux->pre = Banda->deget->pre;
       Banda->deget->pre = aux;
       aux->urm = Banda->deget;
       Banda->deget = aux;
    }
}

void INSERT_RIGHT(TBanda Banda, char c){
    TLista2 aux=AlocCelula2(c);
    aux->pre = Banda->deget;
    aux->urm = Banda->deget->urm;
    if(Banda->deget->urm){ //daca degetul nu este pe ultima pozitie
        Banda->deget->urm->pre = aux; //legam pozitia urmatoare de aux
    }
    Banda->deget->urm = aux; 
    Banda->deget = aux; //mutam degetul pe aux
}

void SHOW_CURRENT(TBanda Banda, FILE * out){
    fprintf(out, "%c\n", Banda->deget->info);
}


void SHOW(TBanda Banda, FILE* out){
    TLista2 aux = Banda->list->urm; //aux este pointerul care parcurge banda
    while(aux != NULL){
        if(aux == Banda->deget){ //daca aux este pe pozitia curenta
            fprintf(out, "|%c|", aux->info);
        }
        else{
            fprintf(out, "%c", aux->info); 
        }
        aux = aux->urm;
    }
    fprintf(out, "\n");
}

//functie pentru operatia UNDO
void UNDO(TStiva *undoStack, TStiva *redoStack, TBanda Banda) {
    if (*undoStack != NULL) {
        Push(redoStack, Banda->deget); // adaugam pointerul la pozitia curenta in stiva pentru REDO
        Banda->deget = Pop(undoStack); // extragem pointerul din varful stivei si il setam ca pozitie curenta
    }
}

//functie pentru operatia REDO
void REDO(TStiva *undoStack, TStiva *redoStack, TBanda Banda) {
    if (*redoStack != NULL) {
        Push(undoStack, Banda->deget); // adaugam pointerul la pozitia curenta in stiva pentru UNDO
        Banda->deget = Pop(redoStack); // extragem pointerul din varful stivei si il setam ca pozitie curenta
    }
}

void EXECUTE(TBanda Banda, TCoada* c, TStiva* undoStack, FILE *out){
	//cautam mereu la inceputul cozii si executam operatia gasita
        if(strcmp(c->inc->info,"MOVE_LEFT") == 0){ 
            MOVE_LEFT(Banda, undoStack);
        }
        else if(strcmp(c->inc->info, "MOVE_RIGHT") == 0){
            MOVE_RIGHT(Banda, undoStack);
        }
        else if(strstr(c->inc->info, "WRITE ") != 0){
            WRITE(Banda, c->inc->info[WRITECHAR], undoStack);
        }
        else if(strstr(c->inc->info,"MOVE_RIGHT_CHAR ") != 0){
            MOVE_RIGHT_CHAR(Banda, c->inc->info[MOVERIGHTCHAR]);
        }
        else if(strstr(c->inc->info,"MOVE_LEFT_CHAR ") != 0){
            MOVE_LEFT_CHAR(Banda, c->inc->info[MOVELEFTCHAR], out);
        }
        else if(strstr(c->inc->info, "INSERT_RIGHT ") != 0){
            INSERT_RIGHT(Banda, c->inc->info[INSERTRIGHT]);
        }
        else if(strstr(c->inc->info, "INSERT_LEFT ") != 0){
            INSERT_LEFT(Banda, c->inc->info[INSERTLEFT], out);
        }

        TCelula *firstCell = c->inc; //salvez adresa primei celule din coada 
        if(c->inc == c->sf) { //daca coada are un singur element
            c->sf = NULL; //coada devine vida
        }
        c->inc = c->inc->urm; //coada se muta cu o pozitie in dreapta
        free(firstCell);
}


void citireFisier(){

    FILE* inputFile = fopen("tema1.in", "r");
    FILE* outputFile = fopen("tema1.out", "w");
    int nrDeOperatii = 0;
    int i = 0;
    TBanda Banda = malloc(sizeof(struct banda)); //alocam memorie pentru banda
    Banda->deget = NULL; //initializam degetul cu NULL
    Banda->list = InitLista2(); //initializam lista
    
    //Adaugam # pe prima pozitie din banda
    Banda->list->urm = AlocCelula2('#');
    Banda->list->urm->pre = Banda->list;
    Banda->deget = Banda->list->urm;

    TCoada* c = InitQ(); //initializam coada
    TStiva undoStack = NULL;
    TStiva redoStack = NULL; 

    if (inputFile == NULL || outputFile == NULL) {
        printf("Error opening files.\n");
        return;
    }

    char buffer[DIMINPUT];
    fscanf(inputFile, "%d", &nrDeOperatii);
    fgets(buffer, DIMINPUT, inputFile);
    
    for(i = 0; i < nrDeOperatii; i++){
        fgets(buffer, DIMINPUT, inputFile);
        if(buffer[strlen(buffer) - 1] == '\n'){
            buffer[strlen(buffer) - 1] = 0; //eliminam \n din buffer
        }
        if(strstr(buffer, "WRITE")!=0){ 
            IntrQ(c, buffer); //adaugam in coada operatia WRITE
        }
        else if(strstr(buffer, "MOVE_LEFT")!=0 && strstr(buffer, "MOVE_LEFT ")==0){
            IntrQ(c, buffer); //adaugam in coada operatia MOVE_LEFT
        }
        else if(strstr(buffer, "MOVE_RIGHT")!=0 && strstr(buffer, "MOVE_RIGHT ")==0){
            IntrQ(c, buffer); //adaugam in coada operatia MOVE_RIGHT
        }
        else if(strstr(buffer,"MOVE_LEFT_CHAR")!=0 ){
            IntrQ(c, buffer);  
        }
        else if(strstr(buffer, "MOVE_RIGHT_CHAR")!=0){
            IntrQ(c, buffer);
        }
        else if(strstr(buffer, "INSERT_LEFT")!=0){
            IntrQ(c, buffer);
        }
        else if(strstr(buffer, "INSERT_RIGHT")!=0){
            IntrQ(c, buffer);
        }
        else if(strstr(buffer, "EXECUTE")!=0){
            EXECUTE(Banda, c, &undoStack, outputFile); 
        }
        else if(strcmp(buffer, "SHOW")==0){
            SHOW(Banda, outputFile);
        }
        else if(strcmp(buffer, "SHOW_CURRENT")==0){
            SHOW_CURRENT(Banda, outputFile);
        }
        else if(strcmp(buffer, "UNDO")==0){
            UNDO(&undoStack, &redoStack, Banda);
        }
        else if(strcmp(buffer, "REDO")==0){
            REDO(&undoStack, &redoStack, Banda);
        }
    }

    DistrQ(&c);
    DistrugereS(&undoStack);
    DistrugereS(&redoStack);
    DistrugeLista2(&Banda);
    fclose(inputFile);
    fclose(outputFile);
}

int main(){
    citireFisier();
    return 0;
}
