import os

import pygame

from hangman import game_session as hangman_session
from sudoku import game_session as sudoku_session


class GameLauncher:
    def __init__(self):
        pygame.init()

        self.running = True

        self.screen = pygame.display.set_mode((800, 800))
        self.load_main_menu()

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.background1 = pygame.image.load(os.path.join(os.path.dirname(__file__), 'hangman_thumbnail.png'))
        self.background2 = pygame.image.load(os.path.join(os.path.dirname(__file__), 'sudoku_thumbnail.jpg'))

        self.button1_rect = pygame.Rect(250, 150, 300, 200)
        self.button2_rect = pygame.Rect(250, 450, 300, 200)

    def run(self):
        while self.running:
            self.screen.fill((100, 100, 100))
            self.draw_buttons()

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle Sudoku choice.
                    if self.button2_rect.collidepoint(event.pos):
                        sudoku_game = sudoku_session.GameSession(self.screen)

                        if sudoku_game.run() == 1:
                            self.running = False
                            break
                        else:
                            self.load_main_menu()
                            continue

                    # Handle Hangman choice.
                    if self.button1_rect.collidepoint(event.pos):
                        hangman_session.initialize_game_session()
                        hangman_session.run_game_session(self.screen)

                        self.load_main_menu()

    @staticmethod
    def load_main_menu():
        menu_icon = pygame.image.load(os.path.join(os.path.dirname(__file__), 'menu_icon.png'))
        pygame.display.set_icon(menu_icon)

        pygame.display.set_caption("Game Launcher")

    def draw_buttons(self):
        # Draw button 1 rectangle
        pygame.draw.rect(self.screen, (52, 152, 219), self.button1_rect)

        # Background image for button 1
        background1_scaled = pygame.transform.scale(self.background1,
                                                    (self.button1_rect.width, self.button1_rect.height))
        self.screen.blit(background1_scaled, (self.button1_rect.x, self.button1_rect.y))

        # Draw button 2 rectangle
        pygame.draw.rect(self.screen, (46, 204, 113), self.button2_rect)

        # Background image for button 2
        background2_scaled = pygame.transform.scale(self.background2,
                                                    (self.button2_rect.width, self.button2_rect.height))
        self.screen.blit(background2_scaled, (self.button2_rect.x, self.button2_rect.y))
