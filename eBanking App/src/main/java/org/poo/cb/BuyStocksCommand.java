package org.poo.cb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import static org.poo.cb.Main.stockFile;

public class BuyStocksCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        cumparaActiuni(splitCommand);
    }

    //Fct pt a extrage din fisierul csv cel mai recent pret al unei actiuni
    private static double getPriceForCompany(String companyName, String stockFile) {
        try (BufferedReader br = new BufferedReader(new FileReader(stockFile))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] splitLine = line.split(",");
                if (splitLine[0].equals(companyName)) {
                    //Ultima coloana
                    return Double.parseDouble(splitLine[splitLine.length - 1]);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return 0.0;
    }

    private static void cumparaActiuni(String[] comanda) {
        String email = comanda[2].trim().toLowerCase();
        String numeCompanie = comanda[3].trim();
        int numarActiuni = Integer.parseInt(comanda[4]);

        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            //Verific daca userul este premium
            boolean isPremium = user.isPremium();

            //Verific daca exista suficienti bani in contul USD
            Cont userContUSD = user.getConturi().get("USD");
            if (userContUSD != null) {
                double pretActiune = getPriceForCompany(numeCompanie, stockFile);

                //Reducere de 5% pt utilizator premium
                if (isPremium) {
                    pretActiune *= 0.95;
                }

                if (userContUSD.getSuma() >= numarActiuni * pretActiune) {
                    userContUSD.adaugaBani(-numarActiuni * pretActiune);
                    user.adaugaActiuni(numeCompanie, numarActiuni);
                } else {
                    System.out.println("Insufficient amount in account for buying stock");
                }
            } else {
                System.out.println("USD account not found for user " + email);
            }
        } else {
            System.out.println("User not found with email " + email);
        }
    }

}
