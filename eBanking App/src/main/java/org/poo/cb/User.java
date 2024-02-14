package org.poo.cb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class User {
    String name;
    String email;
    String prenume;
    String adresa;

    private boolean premium;

    public boolean isPremium() {
        return premium;
    }

    public void setPremium(boolean premium) {
        this.premium = premium;
    }
    List<User> friends;
    private Map<String, Cont> conturi;
    private List<Stock> stocks;

    public User(String name, String email, String prenume, String adresa) {
        this.name = name;
        this.email = email;
        this.prenume = prenume;
        this.adresa = adresa;
        this.friends = new ArrayList<>();
        this.conturi = new HashMap<>();
        this.stocks = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public String getPrenume() {
        return prenume;
    }

    public String getEmail() {
        return email;
    }

    public String getAdresa() {
        return adresa;
    }

    public List<User> getFriends() {
        return friends;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public void setPrenume(String prenume) {
        this.prenume = prenume;
    }

    public void setAdresa(String adresa) {
        this.adresa = adresa;
    }

    public void adaugaPrieten(User friend) {
        friends.add(friend);
    }

    public void adaugaCont(String tipCont) {
        //Verific daca utilizatorul are un cont de acelasi tip
        if (conturi.containsKey(tipCont)) {
            System.out.println("Utilizatorul deja are un cont de tipul " + tipCont);
        } else {
            Cont cont = new Cont(tipCont);
            conturi.put(tipCont, cont);
        }
    }

    public void adaugaBani(String tipCont, double suma) {
        Cont cont = conturi.get(tipCont);
        if (cont != null) {
            cont.adaugaBani(suma);
        } else {
            System.out.println("Contul de tip " + tipCont + " nu exista pentru utilizatorul " + getEmail());
        }
    }


    public List<Stock> getStocks() {
        return stocks;
    }

    public void adaugaActiuni(String numeCompanie, int numarActiuni) {
        Stock stock = new Stock(numeCompanie, numarActiuni);
        stocks.add(stock);
    }

    public String afiseazaPortofoliu() {
        List<String> stocksStr = new ArrayList<>();
        for (Stock stock : stocks) {
            stocksStr.add(String.format("{\"stockname\":\"%s\",\"amount\":%d}", stock.getNumeCompanie(), stock.getNumarActiuni()));
        }

        List<String> conturiStr = new ArrayList<>();
        for (Cont cont : conturi.values()) {
            conturiStr.add(String.format("{\"currencyname\":\"%s\",\"amount\":\"%.2f\"}", cont.getTipValuta(), cont.getSuma()));
        }

        Collections.reverse(conturiStr);

        return String.format("{\"stocks\":[%s],\"accounts\":[%s]}", String.join(",", stocksStr), String.join(",", conturiStr));
    }



    public void exchangeMoney(String fromCurrency, String toCurrency, double amount) {
        Cont fromCont = conturi.get(fromCurrency.toUpperCase());
        Cont toCont = conturi.get(toCurrency.toUpperCase());

        ExchangeRateManager exchangeRateManager = new ExchangeRateManager("src/main/resources/common/exchangeRates.csv");

        //Obtin cursul valutar
        Optional<Double> exchangeRateOp = exchangeRateManager.getExchangeRate(fromCurrency, toCurrency);

        if (exchangeRateOp.isPresent() && fromCont != null && toCont != null) {
            //Verific daca exista bani suficienti in contul sursa
            double exchRate = exchangeRateOp.get();

            if (fromCont.getSuma() - amount * exchRate >= 0 && fromCont.getSuma() > 0 )  {
                double thresholdAmount = 0.5 * fromCont.getSuma();
                if(premium == true){
                    fromCont.adaugaBani(-amount * exchRate);
                    toCont.adaugaBani(amount);
                }
                else {
                    //Comisionul de 1% la sume ce depasesc 50%
                    if (amount * exchRate >= thresholdAmount) {
                        double commission = 0.01 * amount * exchRate;
                        fromCont.adaugaBani(-amount * exchRate - commission);
                        toCont.adaugaBani(amount);
                    } else {
                        fromCont.adaugaBani(-amount * exchRate);
                        toCont.adaugaBani(amount);
                    }
                }
            } else {
                System.out.println("Insufficient amount in account " + fromCurrency + " for exchange");
            }
        } else {
            System.out.println("Invalid currency specified or exchange rate not available.");
        }
    }

    public Map<String, Cont> getConturi() {
        return conturi;
    }


}











