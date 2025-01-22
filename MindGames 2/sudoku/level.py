from sudoku import constants


class Level:
    def __init__(self):
        self.value = 0
        self.hint_cnt = constants.HINTS_NR_LEVEL_0

    # Return 0 if there are levels left, 1 if all levels have been completed.
    def change_to_next_level(self):
        self.value += 1

        match self.value:
            case 0:
                self.hint_cnt = constants.HINTS_NR_LEVEL_0
            case 1:
                self.hint_cnt = constants.HINTS_NR_LEVEL_1
            case 2:
                self.hint_cnt = constants.HINTS_NR_LEVEL_2
            case 3:
                self.hint_cnt = constants.HINTS_NR_LEVEL_3
            case 4:
                return 1

        return 0
