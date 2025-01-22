import os

import pygame

from sudoku import button
from sudoku import constants


class InitialMenu:
    def __init__(self, screen):
        self.screen = screen
        self.running = True

        self.play_button = button.Button(80, 375, 120, 50, constants.PURPLE, "Play")
        self.exit_button = button.Button(300, 375, 120, 50, constants.PURPLE, "Exit")
        self.main_menu_button = button.Button(520, 375, 200, 50, constants.PURPLE, "Main Menu")
        self.button_font = pygame.font.Font("freesansbold.ttf", 35)

    # Return 0 to start playing the game, 1 to exit, 2 to return to main menu.
    def run(self):
        background_path = os.path.join(os.path.dirname(__file__), 'menu_image.jpg')
        background = pygame.image.load(background_path)

        welcome_message_font = pygame.font.Font("freesansbold.ttf", 35)
        welcome_message1 = welcome_message_font.render("Welcome to Sudoku!",
                                                       True, constants.DEEP_PURPLE)
        welcome_message2 = welcome_message_font.render("Click on <Play> to start the game.",
                                                       True, constants.DEEP_PURPLE)

        while self.running:
            self.screen.blit(background, (0, 0))
            x1 = (constants.TOTAL_WIDTH - welcome_message1.get_width()) / 2
            x2 = (constants.TOTAL_WIDTH - welcome_message2.get_width()) / 2
            self.screen.blit(welcome_message1, (x1, constants.WELCOME_MESSAGE_Y))
            self.screen.blit(welcome_message2, (x2, constants.WELCOME_MESSAGE_Y + welcome_message1.get_height() + 2))

            self.play_button.draw(self.screen, self.button_font)
            self.exit_button.draw(self.screen, self.button_font)
            self.main_menu_button.draw(self.screen, self.button_font)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 1

                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_position = pygame.mouse.get_pos()

                    if self.exit_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 1

                    if self.main_menu_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 2

                    if self.play_button.mouse_is_on_button(mouse_position):
                        self.running = False
                        return 0
