# Turing Machine Implementation in C #

### Formally, the Turing Machine comprises two essential elements: ###
- An infinite tape consisting of an infinite set of cells in which characters are written, initialized to a particular character (usually the character "#").
- The read/write head: it moves, according to well-defined instructions, on the tape in 3 positions; to the left (L), to the right (R), or it remains stationary (S).

### There are 4 types of operations: ###
- UPDATE operations:
  - MOVE_LEFT
  - MOVE_RIGHT
  - MOVE_LEFT_CHAR
  - MOVE_RIGHT_CHAR
  - WRITE
  - INSERT_LEFT_CHAR
  - INSERT_RIGHT_CHAR
- QUERY operations:
  - SHOW_CURRENT
  - SHOW
- UNDO/REDO operations
- EXECUTE operation - it is implemented with a tail and a linked list.

#### There will be those comands read from the keyboard. The UPDATE operations will be pushed in a queue as they are read, and the operation EXECUTE will pop one by one these UPDATE operations and will execute them. ####
I implemented functions for reach operation type and for each operation in particular. For code modularization, I used header files in which I declared all the structures I need (stack, queue, lists), in order to have a short and easy to understand main function. I also used different files for basic functions such as push/pop and other that were necessary for the data structures I used.



