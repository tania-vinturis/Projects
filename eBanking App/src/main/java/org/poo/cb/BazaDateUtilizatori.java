package org.poo.cb;

import java.util.HashMap;
import java.util.Map;

public class BazaDateUtilizatori {
    private static BazaDateUtilizatori instance;

    private final Map<String, User> bazaDateUtilizatori = new HashMap<>();

    //Private constructor
    private BazaDateUtilizatori() {
    }

    //Fct pentru obtinerea instantei unice
    public static synchronized BazaDateUtilizatori getInstance() {
        if (instance == null) {
            instance = new BazaDateUtilizatori();
        }
        return instance;
    }

    public Map<String, User> getBazaDateUtilizatori() {
        return bazaDateUtilizatori;
    }

}

