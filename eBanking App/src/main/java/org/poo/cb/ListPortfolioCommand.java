package org.poo.cb;

import java.util.Map;

public class ListPortfolioCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        afiseazaPortofoliu(splitCommand[2]);
    }

    private static void afiseazaPortofoliu(String email) {
        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            System.out.println(user.afiseazaPortofoliu());
        } else {
            System.out.println("Utilizatorul cu email-ul " + email + " nu a fost gasit.");
        }
    }
}
