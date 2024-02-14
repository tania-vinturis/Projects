package org.poo.cb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Recommender {
    public static List<String> recommendStocks(String stockFile) {
        List<String> stocksToBuy = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(stockFile))) {
            br.readLine(); //Dau skip la primul rand
            String line;
            while ((line = br.readLine()) != null) {
                String[] splitLine = line.split(",");
                List<Double> prices = new ArrayList<>(); //Lista pt preturile de pe linia curenta

                //Iterez prin lista de preturi si le adun
                for (int i = 1; i < splitLine.length; i++) {
                    prices.add(Double.parseDouble(splitLine[i]));
                }

                //Calculez SMA pt linia curenta
                double shortTermSMA = calculateSMA(prices.subList(prices.size() - 5, prices.size()));
                double longTermSMA = calculateSMA(prices.subList(1, prices.size()));

                //Verific conditia SMA Crossover si adaug actiunea la lista de recomandari
                if (shortTermSMA >= longTermSMA) {
                    stocksToBuy.add(splitLine[0]);
                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return stocksToBuy;
    }


    //Fct pt media aritmetica
    private static double calculateSMA(List<Double> values) {
        double sum = 0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.size();
    }
}
