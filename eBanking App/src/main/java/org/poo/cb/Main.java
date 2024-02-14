package org.poo.cb;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import static org.poo.cb.BankingFacade.commandStrategies;

public class Main {

    private static void citesteComenzi(String caleFisier, BankingFacade bankingFacade) {
        try (BufferedReader br = new BufferedReader(new FileReader(caleFisier))) {
            String linie;
            while ((linie = br.readLine()) != null) {
                executeCommand(linie, bankingFacade);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void executeCommand(String commandLine, BankingFacade bankingFacade) {
        String[] splitCommand = commandLine.split(" ");
        String commandKey = splitCommand[0] + " " + splitCommand[1];

        CommandStrategy strategy = BankingFacade.getCommandStrategy(commandKey);
        if (strategy != null) {
            strategy.execute(commandLine, bankingFacade.getUserFactory());
        } else {
            System.out.println("Comanda necunoscuta: " + commandLine);
        }
    }
    public static String stockFile;

    public static void main(String[] args) {

        if (args == null || args.length != 3) {
            System.out.println("Running Main");
            return;
        }

        UserFactory userFactory = new DefaultUserFactory();
        String commandsFile = "src/main/resources/" + args[2];
        stockFile = "src/main/resources/" + args[1];

        BankingFacade bankingFacade = new BankingFacade(userFactory);
        BazaDateUtilizatori.getInstance().getBazaDateUtilizatori().clear();
        citesteComenzi(commandsFile, bankingFacade);


    }
}