import os

import pygame

from sudoku import between_levels_menu
from sudoku import button
from sudoku import constants
from sudoku import level
from sudoku import sudoku_sample


class GameRunner:
    def __init__(self, screen):
        self.running = True
        self.screen = screen

        # For game logic.
        self.level = level.Level()
        self.sudoku = sudoku_sample.SudokuSample()
        self.selected_row = 0
        self.selected_column = 0
        self.is_square_selected = False
        self.level_passed = False

        # For displaying the sudoku digits.
        self.digitX = constants.BACKGROUND_START_X + constants.DIGIT_OFFSET
        self.digitY = constants.BACKGROUND_START_Y + constants.DIGIT_OFFSET
        self.digit_font = pygame.font.Font("freesansbold.ttf", 30)
        self.hint_level_font = pygame.font.Font("freesansbold.ttf", 40)

        # For buttons.
        self.hint_button = button.Button(constants.HINT_BUTTON_X, constants.HINT_BUTTON_Y,
                                         120, 50, constants.PURPLE, "Hint")
        self.button_font = pygame.font.Font("freesansbold.ttf", 35)

    # Return 0 to continue playing the game, 1 to exit, 2 to go to main menu.
    def run(self):
        background_path = os.path.join(os.path.dirname(__file__), 'grid.jpg')
        background = pygame.image.load(background_path)

        self.sudoku.load_sudoku(self.level.value)

        while self.running:
            if self.level_passed:
                # Check if all levels have been passed. If so, return success.
                if self.level.change_to_next_level() == 1:
                    self.running = False
                    return 0

                finish_level_rc = self.finish_level()
                if finish_level_rc == 1:
                    self.running = False
                    return 1
                elif finish_level_rc == 2:
                    self.running = False
                    return 2

            self.screen.fill(constants.BLACK)
            self.screen.blit(background, (constants.BACKGROUND_START_X, constants.BACKGROUND_START_Y))

            self.display_sudoku()
            self.hint_button.draw(self.screen, self.button_font)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return 1

                if event.type == pygame.MOUSEBUTTONUP:
                    mouse_position = pygame.mouse.get_pos()

                    if self.hint_button.mouse_is_on_button(mouse_position):
                        self.sudoku.generate_hint(self.level)
                        self.level_passed = self.sudoku.check_winning()
                    else:
                        (self.selected_row, self.selected_column) = self.get_square(mouse_position)
                        if (0 <= self.selected_row < constants.ROWS_CNT
                                and 0 <= self.selected_column < constants.COLUMNS_CNT):
                            self.is_square_selected = True

                if (event.type == pygame.KEYDOWN and self.is_square_selected
                        and self.sudoku.is_square_modifiable(self.selected_row, self.selected_column)):
                    pressed_key = self.get_pressed_key(event)
                    if 0 <= pressed_key <= 9:
                        self.sudoku.riddle[self.selected_row][self.selected_column] = pressed_key

                    if self.sudoku.check_user_input_validity(self.selected_row, self.selected_column):
                        self.level_passed = self.sudoku.check_winning()

                # Navigate with TAB.
                if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                    (self.selected_row, self.selected_column) = self.sudoku.select_next_free_square(self.selected_row,
                                                                                                    self.selected_column,
                                                                                                    self.is_square_selected)
                    self.is_square_selected = True

                # Navigate with left and right arrows.
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    (self.selected_row, self.selected_column) = self.sudoku.select_next_free_square(self.selected_row,
                                                                                                    self.selected_column,
                                                                                                    self.is_square_selected)
                    self.is_square_selected = True

                if event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    (self.selected_row, self.selected_column) = self.sudoku.select_prev_free_square(self.selected_row,
                                                                                                    self.selected_column,
                                                                                                    self.is_square_selected)
                    self.is_square_selected = True

            if self.sudoku.is_square_modifiable(self.selected_row, self.selected_column) and self.is_square_selected:
                self.draw_selection(self.selected_row, self.selected_column)

            self.display_level_number(self.level.value)
            self.display_hint_number(self.level.hint_cnt)
            pygame.display.update()

        return 0

    # Displays the content of the currently loaded sudoku riddle.
    def display_sudoku(self):
        self.digitX = constants.BACKGROUND_START_X + constants.DIGIT_OFFSET
        self.digitY = constants.BACKGROUND_START_Y + constants.DIGIT_OFFSET

        for i in range(0, constants.ROWS_CNT):
            for j in range(0, constants.COLUMNS_CNT):
                if self.sudoku.riddle[i][j] != 0:
                    color = self.sudoku.get_digit_color(i, j)
                    digit = self.digit_font.render(str(self.sudoku.riddle[i][j]), True, color)
                    self.screen.blit(digit, (self.digitX, self.digitY))

                self.digitX = self.digitX + constants.SQUARE_LEN

            self.digitX = constants.BACKGROUND_START_X + constants.DIGIT_OFFSET
            self.digitY += constants.SQUARE_LEN

    # Displays the number of hints left.
    def display_hint_number(self, value):
        hint_message = self.hint_level_font.render("Hints left: " + str(value), True, constants.PURPLE)
        pygame.draw.rect(self.screen, constants.BLACK, [constants.HINT_NR_X, constants.HINT_NR_Y, 100, 50])
        self.screen.blit(hint_message, (constants.HINT_NR_X, constants.HINT_NR_Y))

    # Displays the current level.
    def display_level_number(self, value):
        level_message = self.hint_level_font.render("Level " + str(value) + "/3", True, constants.PURPLE)
        pygame.draw.rect(self.screen, constants.BLACK, [constants.LEVEL_NR_X, constants.LEVEL_NR_Y, 100, 50])
        self.screen.blit(level_message, (constants.LEVEL_NR_X, constants.LEVEL_NR_Y))

    # Returns the grid coordinates of the square clicked by the mouse.
    def get_square(self, mouse_position):
        if not (constants.BACKGROUND_START_X <= mouse_position[0]
                <= constants.BACKGROUND_START_X + constants.BACKGROUND_DIMENSION):
            return self.sudoku.select_next_free_square(0, -1, False)
        if not (constants.BACKGROUND_START_Y <= mouse_position[1]
                <= constants.BACKGROUND_START_Y + constants.BACKGROUND_DIMENSION):
            return self.sudoku.select_next_free_square(0, -1, False)

        x = int((mouse_position[1] - constants.BACKGROUND_START_Y) / constants.SQUARE_LEN)
        y = int((mouse_position[0] - constants.BACKGROUND_START_X) / constants.SQUARE_LEN)

        return x, y

    # Returns the value of the pressed key, if it is a number key.
    # Returns 0, if it is a deletion key.
    # Returns -1, if it is another key.
    def get_pressed_key(self, event):
        match event.key:
            case pygame.K_BACKSPACE:
                return 0
            case pygame.K_DELETE:
                return 0
            case pygame.K_1:
                return 1
            case pygame.K_2:
                return 2
            case pygame.K_3:
                return 3
            case pygame.K_4:
                return 4
            case pygame.K_5:
                return 5
            case pygame.K_6:
                return 6
            case pygame.K_7:
                return 7
            case pygame.K_8:
                return 8
            case pygame.K_9:
                return 9

        return -1

    # Draw selection borders.
    def draw_selection(self, row, column):
        x = constants.BACKGROUND_START_X + column * constants.SQUARE_LEN
        y = constants.BACKGROUND_START_Y + row * constants.SQUARE_LEN
        pygame.draw.line(self.screen, constants.PURPLE, (x + 1, y + 1),
                         (x + constants.SQUARE_LEN - 2, y + 1), 2)
        pygame.draw.line(self.screen, constants.PURPLE, (x + 1, y + 1),
                         (x + 1, y + constants.SQUARE_LEN - 2), 2)
        pygame.draw.line(self.screen, constants.PURPLE, (x + constants.SQUARE_LEN - 2, y + 1),
                         (x + constants.SQUARE_LEN - 2, y + constants.SQUARE_LEN - 2), 2)
        pygame.draw.line(self.screen, constants.PURPLE, (x + 1, y + constants.SQUARE_LEN - 2),
                         (x + constants.SQUARE_LEN - 2, y + constants.SQUARE_LEN - 2), 2)

    # Simulates the menu displayed after finishing a level.
    # Return 0 if the thing goes on.
    # Return 1 if the user exits the game.
    # Return 2 to go back to main menu.
    def finish_level(self):
        level_menu = between_levels_menu.BetweenLevelsMenu(self.screen)

        user_choice = level_menu.run()

        if user_choice == 1:
            return 1
        elif user_choice == 2:
            return 2

        # Load the next Level.
        self.level_passed = False

        self.sudoku = sudoku_sample.SudokuSample()
        self.sudoku.load_sudoku(self.level.value)
        self.screen.fill(constants.BLACK)

        return 0
