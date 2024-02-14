package org.poo.cb;

import java.util.Map;

public class ExchangeMoneyCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        exchangeMoney(splitCommand[2].trim().toLowerCase(), splitCommand[3].trim().toLowerCase(), splitCommand[4].trim().toLowerCase(), Double.parseDouble(splitCommand[5]));
    }

    private static void exchangeMoney(String email, String fromCurrency, String toCurrency, double amount) {

        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            user.exchangeMoney(fromCurrency, toCurrency, amount);
        } else {
            System.out.println("Utilizatorul cu email-ul " + email + " nu a fost gasit.");
        }
    }
}