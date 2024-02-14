package org.poo.cb;

public class BuyPremiumCommand implements CommandStrategy{
    @Override
    public void execute(String commandLine, UserFactory userFactory) {
        String[] splitCommand = commandLine.split(" ");
        String email = splitCommand[2].trim().toLowerCase();

        //Verific daca userul exista in baza de date
        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user == null) {
            System.out.println("User with " + email + " doesn’t exist");
            return;
        }

        //Verific daca utilizatorul are bani in cont pt a cumpara optiunea premium
        Cont userUSDCont = user.getConturi().get("USD");
        if (userUSDCont == null || userUSDCont.getSuma() < 100) {
            System.out.println("Insufficient amount in account for buying premium option");
            return;
        }

        userUSDCont.adaugaBani(-100); //Taxa pt premium
        user.setPremium(true);
    }
}