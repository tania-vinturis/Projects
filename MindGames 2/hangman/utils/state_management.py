import string

import hangman.services.word_service as word_service
from hangman.utils.singleton_meta import SingletonMeta


class StateStore(metaclass=SingletonMeta):
    def __init__(self):
        self._hangman_status: int = 0
        self._clear_word = ""
        self._guessed_word = ""
        self._alphabet_status = {}
        self._game_status = 0
        for letter in string.ascii_uppercase:
            self._alphabet_status[letter] = 0

    def reset_state(self):
        self._hangman_status: int = 0
        self._clear_word = ""
        self._guessed_word = ""
        self._alphabet_status = {}
        self._game_status = 0
        for letter in string.ascii_uppercase:
            self._alphabet_status[letter] = 0
        word_service.choose_word()

    @property
    def game_status(self):
        return self._game_status

    @game_status.setter
    def game_status(self, value):
        self._game_status = value

    @property
    def clear_word(self):
        return self._clear_word

    @clear_word.setter
    def clear_word(self, new_word: str):
        self._clear_word = new_word

    @property
    def guessed_word(self):
        return self._guessed_word

    @guessed_word.setter
    def guessed_word(self, new_word: str):
        self._guessed_word = new_word

    def get_letter_status(self, letter):
        return self._alphabet_status[letter]

    def set_letter_status(self, letter, status):
        self._alphabet_status[letter] = status

    @property
    def hangman_status(self) -> int:
        return self._hangman_status

    def increment_status(self):
        if self._hangman_status < 6:
            self._hangman_status += 1

    @hangman_status.setter
    def hangman_status(self, value):
        self._hangman_status = value
