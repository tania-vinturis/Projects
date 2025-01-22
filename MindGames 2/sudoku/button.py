import pygame

from sudoku import constants


class Button:
    def __init__(self, x, y, width, height, color, text):
        self.x = x
        self.y = y
        self.color = color
        self.text = text
        self.width = width
        self.height = height

    # Displays the button on the screen.
    def draw(self, screen, font):
        pygame.draw.rect(screen, self.color, [self.x, self.y, self.width, self.height])
        message = font.render(self.text, True, constants.WHITE)

        # Centering text inside button.
        offset_x = (self.width - message.get_width()) / 2
        offset_y = (self.height - message.get_height()) / 2

        # Button borders.
        pygame.draw.line(screen, constants.WHITE, (self.x, self.y),
                         (self.x + self.width, self.y), 3)
        pygame.draw.line(screen, constants.WHITE, (self.x, self.y),
                         (self.x, self.y + self.height), 3)
        pygame.draw.line(screen, constants.WHITE, (self.x + self.width, self.y),
                         (self.x + self.width, self.y + self.height), 3)
        pygame.draw.line(screen, constants.WHITE, (self.x, self.y + self.height),
                         (self.x + self.width, self.y + self.height), 3)
        screen.blit(message, (self.x + offset_x, self.y + offset_y + 2))

    # Return true if the mouse clicked on the button, false otherwise.
    def mouse_is_on_button(self, mouse_position):
        if (self.x <= mouse_position[0] <= self.x + self.width
                and self.y <= mouse_position[1] <= self.y + self.height):
            return True
        return False
