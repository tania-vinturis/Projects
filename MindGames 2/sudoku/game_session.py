import os

import pygame

from sudoku import end_game_menu
from sudoku import game_runner
from sudoku import initial_menu


class GameSession:
    def __init__(self, screen):
        pygame.init()

        self.running = True

        self.screen = screen

        # Title and icon
        pygame.display.set_caption("Sudoku")
        icon_path = os.path.join(os.path.dirname(__file__), 'sudoku_icon.png')
        icon = pygame.image.load(icon_path)
        pygame.display.set_icon(icon)

    # Runs the game.
    # Return 1 to exit, 0 to return to main menu.
    def run(self):
        while self.running:
            start_menu = initial_menu.InitialMenu(self.screen)
            start_choice = start_menu.run()
            if start_choice == 1:
                self.running = False
                return 1
            elif start_choice == 2:
                self.running = False
                return 0

            main_game = game_runner.GameRunner(self.screen)
            main_game_rc = main_game.run()
            if main_game_rc == 1:
                self.running = False
                return 1
            elif main_game_rc == 2:
                self.running = False
                return 0

            end_menu = end_game_menu.EndGameMenu(self.screen)
            if end_menu.run() == 1:
                self.running = False
                return 1
