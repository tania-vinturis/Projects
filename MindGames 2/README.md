

# MindGames

---

## Table of contents
1. [What is MindGames?](#what-is-mindgames)
2. [Prerequisites](#prerequisites)
3. [Run procedure](#run-procedure)
4. [Playing the games](#playing-the-games)
    * [Sudoku gameplay](#sudoku-gameplay)
    * [HangMan gameplay](#hangman-gameplay)
5. [Implementation details](#implementation-details)
    * [Sudoku creation](#sudoku-creation)
    * [HangMan creation](#hangman-creation)
6. [Conclusions](#conclusions)

---

## What is MindGames?
* MindGames is a collection of games that stimulate players' problem-solving
capacities.
* It currently consists of two separate games:
  * ***Sudoku***
  * ***HangMan***
* It also has a menu, which allows the user to choose which game he wants to
play.

---

## Prerequisites
* To successfully run the games, it is necessary that the user has python3
installed on his machine, and also pygame and numpy.

---

## Run procedure
* When running the Launcher file, the user will be presented with a menu,
where he can choose which game he wants to play.
* If the user chooses to play Sudoku, he will be presented with a new menu,
described in the [Sudoku gameplay](#sudoku-gameplay) section.
* If the user chooses to play HangMan, he will be presented with a new menu,
described in the [HangMan gameplay](#hangman-gameplay) section.

---

## Playing the games
### Sudoku gameplay
* The game's goal is to complete a 9x9 grid with digits from 1 to 9, without
repeating them on any row, column or 3x3 mini-grid.
* There are 4 levels of difficulty, from 0 to 3.
* Press the `Play` button to start the first level.
* Use the mouse to select a square to place a digit.
* If the digit is already placed on any row, column or mini-grid, it will be
invalid and colored in red, else it will appear blue.
* There are a pre-determined number of hints for every level.
* For grid navigation, `TAB` and `Left and Right Arrows` can be used.

### HangMan gameplay
* The game's goal is to guess the word by guessing the letters that compose it.
* The player has 7 lives, and loses one every time he guesses a wrong letter.
* The player wins if he guesses all the letters before losing all his lives.
* The lives are represented by the components of a hangman, which are drawn one
by one as the player guesses the letters wrong.
* After the player wins or loses, he can choose to play again or to exit the game.

---

## Implementation details
### Sudoku creation
* This game was developed by Marina Oprea and Tudor Zaharia, together.
* We used the `Code with me` feature of PyCharm to simultaneously edit the
project, while on a call on Messenger :).
* It is written fully in Python, using the `pygame` and `numpy` libraries.
* OOP concepts have been used, all functionalities being placed in `classes`.
* `GameSession` is the control class, responsible with linking the
functionalities, while `GameRunner` controls the sudoku game per se.
* Each menu has a different source class, and their respective `run()` method
returns 1 in case the user wants to exit the game, or 0 if the game should
go on.

### HangMan creation
* This game was developed by Alexandra Ioniță and Mihnea Stamatie, together.
* We used the `Code with me` feature of PyCharm to simultaneously edit the
project, while on a call on Discord :).
* It is written fully in Python, using the `pygame` library.
* The hangman package has been used, which contains 3 helper packages:
  * 'assets' - contains the images, colors and fonts used in the game
  * 'services' - contains the logic of giving words to the player and
  drawing the hangman
  * 'utils' - contains the button, global constants, game status
* In the hangman package there are the files that start the game 'main.py' and
the file that contains the game logic 'game_session.py'.

---

## Conclusions
* The main and obvious difficulty was that we had no previous experience with
Python, so thorough documentation and tutorial watching was necessary to get
the thing going, but, overall, it was a fun experience.
* Link to GitHub repo: https://github.com/tudorz23/MindGames.git
