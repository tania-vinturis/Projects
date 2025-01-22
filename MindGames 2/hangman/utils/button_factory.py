from hangman.utils.button_model import ButtonModel


class ButtonFactory:
    _button_instances = {}

    @classmethod
    def get_button(cls, letter):
        if letter not in cls._button_instances:
            cls._button_instances[letter] = ButtonModel(letter)
        return cls._button_instances[letter]
