package org.poo.cb;

import java.util.Map;
import java.util.HashMap;
public class BankingFacade {
    private final UserFactory userFactory;
    static final Map<String, CommandStrategy> commandStrategies = new HashMap<>();

    public BankingFacade(UserFactory userFactory) {
        this.userFactory = userFactory;
        initializeCommandStrategies();
    }

    private void initializeCommandStrategies() {
        commandStrategies.put("CREATE USER", new CreateUserCommand());
        commandStrategies.put("LIST USER", new ListUserCommand());
        commandStrategies.put("ADD FRIEND", new AddFriendCommand());
        commandStrategies.put("ADD ACCOUNT", new AddAccountCommand());
        commandStrategies.put("ADD MONEY", new AddMoneyCommand());
        commandStrategies.put("LIST PORTFOLIO", new ListPortfolioCommand());
        commandStrategies.put("EXCHANGE MONEY", new ExchangeMoneyCommand());
        commandStrategies.put("TRANSFER MONEY", new TransferMoneyCommand());
        commandStrategies.put("BUY STOCKS", new BuyStocksCommand());
        commandStrategies.put("RECOMMEND STOCKS", new RecommendStocks());
        commandStrategies.put("BUY PREMIUM", new BuyPremiumCommand());
    }

    public UserFactory getUserFactory() {
        return userFactory;
    }

    static CommandStrategy getCommandStrategy(String commandKey) {
        return commandStrategies.getOrDefault(commandKey, null);
    }


}
