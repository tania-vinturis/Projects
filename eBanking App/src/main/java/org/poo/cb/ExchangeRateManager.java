package org.poo.cb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class ExchangeRateManager {
    private final Map<String, Map<String, Double>> exchangeRates;

    public ExchangeRateManager(String filePath) {
        this.exchangeRates = readExchangeRatesFromFile(filePath);
    }

    private Map<String, Map<String, Double>> readExchangeRatesFromFile(String filePath) {
        Map<String, Map<String, Double>> rates = new HashMap<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String headerLine = br.readLine();
            String[] currencies = headerLine.split(",");

            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                String fromCurrency = values[0];

                Map<String, Double> rowRates = new HashMap<>();
                for (int colIndex = 1; colIndex < currencies.length; colIndex++) {
                    String toCurrency = currencies[colIndex];
                    double rate = Double.parseDouble(values[colIndex]);
                    rowRates.put(toCurrency, rate);
                }

                rates.put(fromCurrency, rowRates);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rates;
    }

    public Optional<Double> getExchangeRate(String fromCurrency, String toCurrency) {
        //Verific daca map-ul contine toCurrency
        if (exchangeRates.containsKey(toCurrency.toUpperCase())) {
            //Caut valoarea pentru from currency
            Map<String, Double> rowRates = exchangeRates.get(toCurrency.toUpperCase());
            Double rate = rowRates.get(fromCurrency.toUpperCase());

            if (rate != null) {
                return Optional.of(rate);
            }
        }

        System.out.println("Exchange rate not found.");
        return Optional.empty();
    }



}
