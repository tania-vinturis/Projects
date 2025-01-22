import string

import pygame.display
from pygame import Surface

import hangman.assets.images as images
import hangman.services.word_service as word_service
from hangman.assets import fonts as fonts
from hangman.assets.colors import BLACK, WHITE, GREEN, RED
from hangman.assets.fonts import TITLE_FONT, WORD_FONT
from hangman.utils.button_factory import ButtonFactory
from hangman.utils.button_model import ButtonModel
from hangman.utils.global_constants import WIDTH
from hangman.utils.state_management import StateStore

TITLE = "Hangman"
TRY_AGAIN_WIDTH = 400
TRY_AGAIN_HEIGHT = 40
TRY_AGAIN_TOP = 750
TRY_AGAIN_LEFT = round((WIDTH - TRY_AGAIN_WIDTH) / 2)


def draw_title(screen: Surface):
    title = fonts.get_font(TITLE_FONT).render(TITLE, True, BLACK)
    screen.blit(title, (WIDTH / 2 - title.get_width() / 2, 40))


def draw_image(screen: Surface):
    image = images.get_hangman_image(StateStore().hangman_status)
    screen.blit(image, (WIDTH / 2 - image.get_width() / 2, 180))


def draw_word(screen: Surface):
    word = fonts.get_font(WORD_FONT).render(word_service.prepare_word_for_display(), True, BLACK)
    screen.blit(word, (WIDTH / 2 - word.get_width() / 2, 460))


def draw_buttons(screen: Surface):
    gap = 15
    start_x = round((WIDTH - (ButtonModel.BUTTON_WIDTH + gap) * 13 - 3 * gap) / 2)
    start_y = 550
    for i, letter in enumerate(string.ascii_uppercase):
        button = ButtonFactory.get_button(letter)
        button.draw(screen, (start_x + gap * 2 + ((ButtonModel.BUTTON_WIDTH + gap) * (i % 13)),
                             start_y + ((i // 13) * (gap + ButtonModel.BUTTON_HEIGHT))))


def draw_message(screen: Surface):
    message_text = None
    color = None
    match StateStore().game_status:
        case 1:
            message_text = "You Lose!"
            color = RED
        case 2:
            message_text = "You Win!"
            color = GREEN
    if message_text is not None:
        message = fonts.get_font(WORD_FONT).render(message_text, True, color)
        screen.blit(message, (WIDTH / 2 - message.get_width() / 2, 680))


def draw_try_again_button(screen: Surface):
    if StateStore().game_status != 0:
        hit_box = pygame.Rect(TRY_AGAIN_LEFT, TRY_AGAIN_TOP, TRY_AGAIN_WIDTH, TRY_AGAIN_HEIGHT)
        pygame.draw.rect(screen, BLACK, hit_box,
                         3, 10)
        try_again_text = fonts.get_font(fonts.TRY_AGAIN_FONT).render("Press here to try again!", True, BLACK)
        screen.blit(try_again_text, (TRY_AGAIN_LEFT + (TRY_AGAIN_WIDTH - try_again_text.get_width()) / 2,
                                     TRY_AGAIN_TOP + (TRY_AGAIN_HEIGHT - try_again_text.get_height()) / 2))
        return hit_box


def draw(screen: Surface):
    screen.fill(WHITE)
    draw_title(screen)
    draw_image(screen)
    draw_word(screen)
    draw_buttons(screen)
    draw_message(screen)
    try_again_hit_box = draw_try_again_button(screen)
    pygame.display.update()
    return try_again_hit_box
