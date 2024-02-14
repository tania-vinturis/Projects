package org.poo.cb;

public class AddMoneyCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        adaugaBani(splitCommand);
    }

    private static void adaugaBani(String[] comanda) {
        // Adaug bani in contul utilizatorului
        String email = comanda[2].trim().toLowerCase();
        String tipCont = comanda[3].trim().toUpperCase();
        double suma = Double.parseDouble(comanda[4]);

        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            user.adaugaBani(tipCont, suma);
        } else {
            System.out.println("Utilizatorul cu email-ul " + email + " nu a fost gasit.");
        }
    }
}