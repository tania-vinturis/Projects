import pygame

import hangman.assets.fonts as fonts
import hangman.services.word_service as word_service
from hangman.assets.colors import BLACK, RED, GREEN
from hangman.utils.state_management import StateStore


class ButtonModel:
    BORDER_RADIUS = 10
    BORDER_WIDTH = 4

    BUTTON_WIDTH = 45
    BUTTON_HEIGHT = 45

    def __init__(self, letter):
        self.top = None
        self.left = None
        self.letter = letter

    def draw(self, screen, position):
        color = BLACK
        match StateStore().get_letter_status(self.letter):
            case 0:
                color = BLACK
            case 1:
                color = RED
            case 2:
                color = GREEN

        pygame.draw.rect(screen, color, pygame.Rect(position[0], position[1],
                                                    self.BUTTON_WIDTH, self.BUTTON_HEIGHT),
                         self.BORDER_WIDTH, self.BORDER_RADIUS)
        self.top = position[1]
        self.left = position[0]
        letter_text = fonts.get_font(fonts.LETTER_FONT).render(self.letter, True, color)
        screen.blit(letter_text, (position[0] + (self.BUTTON_WIDTH - letter_text.get_width()) / 2,
                                  position[1] + (self.BUTTON_HEIGHT - letter_text.get_height()) / 2))

    def handle_click(self, position):
        if StateStore().get_letter_status(self.letter) != 0:
            return False

        if StateStore().game_status != 0:
            return False

        if self.top is None or self.left is None:
            return False

        if (self.left < position[0] < self.left + self.BUTTON_WIDTH and
                self.top < position[1] < self.top + self.BUTTON_HEIGHT):
            word_service.take_a_guess(self.letter)
            return True

        return False
