# eBanking Application

This project is aimed at developing essential functionalities for an eBanking mobile application. Users can create accounts, manage their funds, make transfers to other users, exchange currencies, and invest in stocks or cryptocurrencies. The application also offers real-time stock recommendations to help users make informed investment decisions.

## The main functionalities include:

- User Account Creation: Users can register with unique email addresses, create currency accounts, and manage friends.
- Fund Management: Users can deposit money, exchange currencies, and transfer funds to friends.
- Investments: Users can buy stocks and cryptocurrencies.
- Recommendations: The application suggests profitable stocks based on a Simple Moving Averages (SMA) crossover strategy.

## Entities

To implement the eBanking functionalities, the following entities are modeled:

### User:
Attributes: email (unique), name, surname, address, and portfolio.
Portfolio: Contains multiple currency accounts, stocks, and a friend list.
Friends: A list of other users to whom funds can be transferred.
### Account:
Attributes: currency type (must be one of the accepted currencies) and balance.
### Stock:
Attributes: company name and daily values for the past 10 days.

## Commands

### User and Account Management Commands
CREATE USER <email> <firstname> <lastname> <address>
Creates a new user with a unique email address.
ADD FRIEND <emailUser> <emailFriend>
Adds a friend to the user’s list, allowing mutual fund transfers.
ADD ACCOUNT <email> <currency>
Creates an account for the user in the specified currency.
ADD MONEY <email> <currency> <amount>
Deposits funds into the user’s specified currency account.
Transaction and Investment Commands
EXCHANGE MONEY <email> <sourceCurrency> <destinationCurrency> <amount>
Converts funds between two of the user’s accounts based on provided exchange rates.
TRANSFER MONEY <email> <friendEmail> <currency> <amount>
Transfers funds from one user to a friend in the specified currency.
BUY STOCKS <email> <company> <noOfStocks>
Purchases stocks in a company with funds from the user’s USD account.
RECOMMEND STOCKS
Provides a list of recommended stocks based on SMA crossover strategy.
Listing Commands
LIST USER <email>
Displays user information in JSON format.
LIST PORTFOLIO <email>
Shows the user’s portfolio, including stocks and currency accounts.
Data Input and Output

## Input Data
Currencies: List of accepted currency types provided in currencies.txt.
Exchange Rates: Rates stored in exchangeRates.csv with conversion data.
Stock Values: Stock prices for the last 10 days stored in stockValues.csv.
Commands: Commands are read from commands.txt.
Output
All command results are displayed in the console.

## Design Patterns

Singleton Pattern: To manage global access to exchange rates or stock values.
Factory Pattern: To create account or stock instances based on user input.
Observer Pattern: For updating users with stock recommendations in real time.
Strategy Pattern: Implemented in the SMA crossover recommendation algorithm.
Note: These patterns should be used thoughtfully, and explanations for each must be documented.

##b Bonus Features

Premium Account
Users can purchase a premium option for $100 (USD). Premium users benefit from:
No commission on currency exchanges.
Discounts on recommended stock purchases (5% off).
