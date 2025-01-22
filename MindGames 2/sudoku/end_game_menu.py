import os

import pygame

from sudoku import button
from sudoku import constants


class EndGameMenu:
    def __init__(self, screen):
        self.screen = screen
        self.running = True

        self.play_again_button = button.Button(280, 300, 240, 50, constants.PURPLE, "Play Again")
        self.exit_button = button.Button(340, 400, 120, 50, constants.PURPLE, "Exit")
        self.button_font = pygame.font.Font("freesansbold.ttf", 35)

    # Return 0 to play again, 1 to exit.
    def run(self):
        background_path = os.path.join(os.path.dirname(__file__), 'menu_image.jpg')
        background = pygame.image.load(background_path)

        message_font = pygame.font.Font("freesansbold.ttf", 35)
        message1 = message_font.render("Thank you for playing!", True, constants.DEEP_PURPLE)
        message2 = message_font.render("You are a sudoku master!", True, constants.DEEP_PURPLE)

        while self.running:
            self.screen.blit(background, (0, 0))
            x1 = (constants.TOTAL_WIDTH - message1.get_width()) / 2
            x2 = (constants.TOTAL_WIDTH - message2.get_width()) / 2
            self.screen.blit(message1, (x1, constants.FINISH_LEVEL_MESSAGE_Y))
            self.screen.blit(message2, (x2, constants.FINISH_LEVEL_MESSAGE_Y + message1.get_height() + 2))

            self.play_again_button.draw(self.screen, self.button_font)
            self.exit_button.draw(self.screen, self.button_font)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 1

                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_position = pygame.mouse.get_pos()

                    if self.play_again_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 0

                    if self.exit_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 1
