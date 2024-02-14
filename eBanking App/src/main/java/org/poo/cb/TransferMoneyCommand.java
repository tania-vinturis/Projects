package org.poo.cb;

public class TransferMoneyCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        transferMoney(splitCommand[2].trim().toLowerCase(), splitCommand[3].trim().toLowerCase(), splitCommand[4].trim().toUpperCase(), Double.parseDouble(splitCommand[5]));
    }


    private static void transferMoney(String email, String friendEmail, String currency, double amount) {
        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        User friend = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(friendEmail);

        if (user != null && friend != null) {
            //Verific daca sunt suficienti bani in cont
            Cont userCont = user.getConturi().get(currency);
            if (userCont != null && userCont.getSuma() >= amount) {
                //Verific daca user si friend sunt prieteni
                if (user.getFriends().contains(friend)) {
                    //Transfer banii
                    userCont.adaugaBani(-amount);
                    friend.adaugaBani(currency, amount);
                } else {
                    System.out.println("You are not allowed to transfer money to " + friendEmail);
                }
            } else {
                System.out.println("Insufficient amount in account " + currency + " for transfer");
            }
        } else {
            System.out.println("Utilizatorul sau prietenul nu au fost găsiți.");
        }
    }
}
