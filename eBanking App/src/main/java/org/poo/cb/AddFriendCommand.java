package org.poo.cb;

import java.util.Map;

public class AddFriendCommand implements CommandStrategy{
    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        adaugaPrieten(splitCommand);
    }

    private static void adaugaPrieten(String[] comanda) {
        String senderEmail = comanda[2].trim().toLowerCase();
        String friendEmail = comanda[3].trim().toLowerCase();

        User sender = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(senderEmail);
        User friend = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(friendEmail);
        if (sender != null && friend != null) {
            sender.adaugaPrieten(friend);
            friend.adaugaPrieten(sender); //Adaug reciproc si userul curent la lista prietenului
        }
    }
}