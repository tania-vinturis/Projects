import os

import pygame

from sudoku import button
from sudoku import constants


class BetweenLevelsMenu:
    def __init__(self, screen):
        self.screen = screen
        self.running = True

        # Buttons
        self.next_level_button = button.Button(340, 300, 120, 50, constants.PURPLE, "Next")
        self.main_menu_button = button.Button(275, 400, 250, 50, constants.PURPLE, "Main Menu")
        self.exit_button = button.Button(340, 500, 120, 50, constants.PURPLE, "Exit")
        self.button_font = pygame.font.Font("freesansbold.ttf", 35)

    # Return 0 to continue playing the game, 1 to exit, 2 to return to main menu.
    def run(self):
        background_path = os.path.join(os.path.dirname(__file__), 'menu_image.jpg')
        background = pygame.image.load(background_path)

        # Message.
        message_font = pygame.font.Font("freesansbold.ttf", 35)
        message1 = message_font.render("Congratulations!",
                                       True, constants.DEEP_PURPLE)
        message2 = message_font.render("You can now access the next level:", True, constants.DEEP_PURPLE)

        while self.running:
            self.screen.fill(constants.BLACK)
            self.screen.blit(background, (0, 0))
            x1 = (constants.TOTAL_WIDTH - message1.get_width()) / 2
            x2 = (constants.TOTAL_WIDTH - message2.get_width()) / 2
            self.screen.blit(message1, (x1, constants.ENDGAME_MESSAGE_Y))
            self.screen.blit(message2, (x2, constants.ENDGAME_MESSAGE_Y + message1.get_height() + 2))

            self.next_level_button.draw(self.screen, self.button_font)
            self.exit_button.draw(self.screen, self.button_font)
            self.main_menu_button.draw(self.screen, self.button_font)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 1

                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_position = pygame.mouse.get_pos()

                    if self.next_level_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 0

                    if self.exit_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 1

                    if self.main_menu_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 2
