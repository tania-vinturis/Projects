import string

import pygame
from pygame import Rect

from hangman.services import draw_service as draw_service
from hangman.services import word_service as word_service
from hangman.utils.button_factory import ButtonFactory
from hangman.utils.state_management import StateStore

GAME_TITLE = "Hangman"


# screen: Surface


def initialize_game_session():
    # pygame.init()

    # global screen
    # screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(GAME_TITLE)
    word_service.choose_word()


def _handle_click_event(position, try_again_hit_box: Rect):
    for letter in string.ascii_uppercase:
        if ButtonFactory.get_button(letter).handle_click(position):
            break

    if try_again_hit_box is not None and try_again_hit_box.collidepoint(position[0], position[1]):
        StateStore().reset_state()


def run_game_session(screen):
    run = True

    while run:
        try_again_hit_box = draw_service.draw(screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                _handle_click_event(pygame.mouse.get_pos(), try_again_hit_box)

    # pygame.quit()
