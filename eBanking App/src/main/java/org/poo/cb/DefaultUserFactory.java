package org.poo.cb;

import java.util.HashMap;
import java.util.Map;

public class DefaultUserFactory implements UserFactory{

    @Override
    public User createUser(String name, String email, String prenume, String adresa) {
        return new User(name, email, prenume, adresa);
    }

}
