package org.poo.cb;

import java.util.List;

import static org.poo.cb.Main.stockFile;
import static org.poo.cb.Recommender.recommendStocks;

public class RecommendStocks implements CommandStrategy{

    @Override
    public void execute(String command, UserFactory uerFactory) {
        List<String> recommendedStocks = recommendStocks(stockFile);
        afiseazaRecomandari(recommendedStocks);
    }
    static void afiseazaRecomandari(List<String> recommendedStocks) {
        System.out.print("{\"stocksToBuy\": [");
        for (int i = 0; i < recommendedStocks.size(); i++) {
            System.out.print("\"" + recommendedStocks.get(i) + "\"");
            if (i < recommendedStocks.size() - 1) {
                System.out.print(", ");
            }
        }
        System.out.println("]}");
    }
}
