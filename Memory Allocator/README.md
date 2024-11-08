# Minimalistic Memory Allocator

## Project Overview

This project implements a basic memory allocator in C, covering core memory management functions: malloc(), calloc(), realloc(), and free(). The custom allocator leverages Linux system calls (brk(), mmap(), and munmap()) to manage memory efficiently while addressing common challenges such as fragmentation and alignment.

## Objectives

- Develop foundational knowledge of memory management by creating basic versions of memory allocation functions.
- Explore Linux memory management syscalls and their usage.
- Address bottlenecks in memory allocation and optimize performance.


## The project includes the following main directories:

- src/: Contains the source code for the allocator (os_malloc, os_calloc, os_realloc, and os_free).
- tests/: Contains the test suite and a Python script (run_tests.py) for validating your implementation.
- utils/: Provides utility files, including osmem.h (library interface) and block_meta.h (metadata for memory blocks).

## API
os_malloc(size_t size): Allocates a memory block of size bytes and returns a pointer.
Uses brk() for allocations below a threshold and mmap() for larger allocations.
Returns NULL if size is 0.
os_calloc(size_t nmemb, size_t size)
Allocates memory for an array of nmemb elements, each size bytes, initializing all bytes to zero.
Follows the same allocation strategy as os_malloc.
Returns NULL if nmemb or size is 0.
*os_realloc(void ptr, size_t size)
Resizes the memory block pointed to by ptr.
Attempts to expand the block if possible.
Coalesces adjacent free blocks if necessary.
Returns NULL if ptr points to a freed block or if the block cannot be resized.
*os_free(void ptr)
Frees the memory block pointed to by ptr.
Marks the block as free, enabling reuse without returning memory to the OS.
Calls munmap() for mapped memory blocks.

## Memory Management Techniques
- Memory Alignment: Ensures that all allocations are aligned to 8 bytes for optimal performance on 64-bit systems.
- Block Reuse and Splitting: Reuses freed blocks where possible; larger blocks are split to minimize wasted memory.
- Block Coalescing: Combines adjacent free blocks to reduce fragmentation, particularly before searching for a block or expanding with os_realloc().
- Best-fit Strategy: Chooses the smallest free block that fits the requested size to reduce fragmentation.
- Heap Preallocation: Allocates a large chunk of memory (128 KB) on first allocation to reduce brk() calls for small allocations.

## Data Structures
block_meta Structure
Manages metadata for each memory block:
struct block_meta {
    size_t size;
    int status;
    struct block_meta *prev;
    struct block_meta *next;
};
- Stores size, status (free or allocated), and pointers to neighboring blocks.
## Building and Running the Project

### Build the Library
Navigate to the src/ directory and run:
make
This command compiles the code and creates libosmem.so, the shared library for the memory allocator.
Run Tests
Use the Python script in the tests/ directory to test the implementation:
python3 run_tests.py
This script runs each test and compares the results with reference outputs to validate syscall accuracy and function performance.
Additional Notes

Error Handling: All syscalls check for errors, and the DIE() macro handles any failures.
Limitations: The allocator avoids mremap() for portability and relies on sbrk() as an alternative to brk().
