import os
import random

from hangman.utils.state_management import StateStore


def choose_word():
    words = []
    file_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'most_common_english_words.txt')
    with open(file_path, 'r') as file:
        for line in file:
            words.append(line)
    rand_index = random.randrange(len(words))
    StateStore().clear_word = words[rand_index][:len(words[rand_index]) - 1].upper()
    StateStore().guessed_word = '_' * len(StateStore().clear_word)


def take_a_guess(letter):
    new_guessed_word = ""
    old_guessed_word = StateStore().guessed_word
    if letter in StateStore().clear_word:
        for i in range(len(StateStore().clear_word)):
            if StateStore().clear_word[i] == letter:
                new_guessed_word += letter
            else:
                new_guessed_word += old_guessed_word[i]
        StateStore().guessed_word = new_guessed_word
        StateStore().set_letter_status(letter, 2)
    else:
        StateStore().set_letter_status(letter, 1)
        StateStore().increment_status()

    if "_" not in StateStore().guessed_word:
        StateStore().game_status = 2

    if StateStore().hangman_status == 6:
        StateStore().game_status = 1
        StateStore().guessed_word = StateStore().clear_word


def prepare_word_for_display():
    processed_word = ""
    for letter in StateStore().guessed_word:
        processed_word += letter + " "
    return processed_word[:len(processed_word) - 1]
