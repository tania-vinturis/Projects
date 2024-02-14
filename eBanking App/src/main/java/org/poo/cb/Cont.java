package org.poo.cb;

public class Cont {
    private final String tipValuta;
    double suma;

    public Cont(String tipValuta) {
        this.tipValuta = tipValuta;
    }

    public void adaugaBani(double suma) {
        this.suma += suma;
    }

    public String getTipValuta() {
        return tipValuta;
    }

    public double getSuma() {
        return suma;
    }

}