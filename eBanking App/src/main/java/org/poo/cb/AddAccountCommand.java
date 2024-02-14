package org.poo.cb;

public class AddAccountCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitComanda = command.split(" ");
        adaugaCont(splitComanda);
    }

    private static void adaugaCont(String[] comanda) {
        String email = comanda[2].trim().toLowerCase();
        String tipCont = comanda[3].trim().toUpperCase();

        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            user.adaugaCont(tipCont);
        } else {
            System.out.println("Utilizatorul cu email-ul " + email + " nu a fost gasit.");
        }
    }
}