package org.poo.cb;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ListUserCommand implements CommandStrategy{

    @Override
    public void execute(String command, UserFactory uerFactory) {
        String[] splitCommand = command.split(" ");
        listUser(splitCommand[2].trim().toLowerCase());
    }

    private static void listUser(String email) {
        User user = BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().get(email);
        if (user != null) {
            System.out.println(formatUser(user));
        } else {
            System.out.println("user with " + email + " doesn't exist");
        }
    }


    private static String formatUser(User user) {
        StringBuilder friendsStringBuilder = new StringBuilder();
        friendsStringBuilder.append('[');
        for (User friend : user.getFriends()) {
            friendsStringBuilder.append('"').append(friend.getEmail()).append('"');
        }
        friendsStringBuilder.append(']');

        return String.format("{\"email\":\"%s\",\"firstname\":\"%s\",\"lastname\":\"%s\",\"address\":\"%s\",\"friends\":%s}",
                user.getEmail(), user.getName(), user.getPrenume(), user.getAdresa(), friendsStringBuilder.toString());
    }
}
