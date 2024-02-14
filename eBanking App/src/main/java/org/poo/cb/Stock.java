package org.poo.cb;

public class Stock {
    private String numeCompanie;
    private int numarActiuni;

    public Stock(String numeCompanie, int numarActiuni) {
        this.numeCompanie = numeCompanie;
        this.numarActiuni = numarActiuni;
    }

    public String getNumeCompanie() {
        return numeCompanie;
    }

    public int getNumarActiuni() {
        return numarActiuni;
    }
}
