package org.poo.cb;

import java.util.Map;

public interface CommandStrategy {
    void execute(String command, UserFactory uerFactory);
}
