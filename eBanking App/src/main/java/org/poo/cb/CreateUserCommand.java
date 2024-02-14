package org.poo.cb;

public class CreateUserCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        adaugaUtilizator(splitCommand, uerFactory);
    }

    private static void adaugaUtilizator(String[] comanda, UserFactory userFactory) {
        String email = comanda[2].trim().toLowerCase();
        String nume = comanda[3];
        String prenume = comanda[4];
        String adresa = comanda[5] + " " + comanda[6] + " " + comanda[7] + " " + comanda[8] + " " + comanda[9] + " " + comanda[10];

        User existingUser = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (existingUser != null) {
            System.out.println("user with " + email + " already exists");
            existingUser.setName(nume);
            existingUser.setPrenume(prenume);
            existingUser.setAdresa(adresa);
        } else {
            User utilizator = userFactory.createUser(nume, email, prenume, adresa);
            BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().put(email, utilizator);
        }
    }
}