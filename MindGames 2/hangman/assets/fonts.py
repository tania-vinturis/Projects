import pygame

TITLE_FONT = ('comicsans', 70)
WORD_FONT = ('comicsans', 40)
LETTER_FONT = ('comicsans', 30)
TRY_AGAIN_FONT = ('comicsans', 20)


def get_font(font_info):
    return pygame.font.SysFont(font_info[0], font_info[1])
