// SPDX-License-Identifier: BSD-3-Clause
#include "osmem.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <assert.h>
#include "block_meta.h"
#define MAP_ANONYMOUS 0x20
#define MMAP_THRESHOLD (128 * 1024)

// functie pt alinierea blocurilor <=> a face un nr multiplu de 8
int ALIGN(int n)
{
	int rest = n % 8;

	if (rest == 0)
		return n;
	return n + (8 - rest);
}

struct block_meta *global_base;

void initList(struct block_meta *Block, size_t size, int status)
{
	Block->size = size;
	Block->status = status;
	Block->prev = NULL;
	Block->next = NULL;
}

struct block_meta *best_block_choice(size_t size)
{
	struct block_meta *current = global_base;
	struct block_meta *best_fit = NULL;
	int found = 0;

	for (; current != NULL; current = current->next) {
		if (current->status == STATUS_FREE && current->size >= size) {
			if (found != 0 && current->size > best_fit->size)
				break;
			best_fit = current;
			found = 1;
		}
	}
	return best_fit;
}

struct block_meta *request_space_malloc(size_t size)
{
	struct block_meta *block;
	void *request;

	if (size < MMAP_THRESHOLD) {
		request = sbrk(size);
		if (request == MAP_FAILED)
			DIE(request, "Error at sbrk in first heap alloc\n");
		block = (struct block_meta *)request;
		initList(block, size, STATUS_ALLOC);
	}	else {
		request = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (request == MAP_FAILED)
			DIE(request, "Error in mmap");
		block = (struct block_meta *)request;
		initList(block, size, STATUS_MAPPED);
	}
	return block;
}

struct block_meta *request_space_calloc(size_t size)
{
	struct block_meta *block;
	void *request;

	if (size < (size_t)sysconf(_SC_PAGE_SIZE)) {
		request = sbrk(size);
		if (request == MAP_FAILED)
			DIE(request, "Error at sbrk in first heap alloc\n");
		block = (struct block_meta *)request;
		initList(block, size, STATUS_ALLOC);
	}	else {
		request = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		if (request == MAP_FAILED)
			DIE(request, "Error in mmap");
		block = (struct block_meta *)request;
		initList(block, size, STATUS_MAPPED);
	}
	return block;
}

void split_block(struct block_meta *block, size_t size)
{
	if (block->size <= size)
		return;
    // Adresa noului bloc
	struct block_meta *new_block = (struct block_meta *)((void *)block + size);
	int left_size = block->size - size;
    // Initializez lista pentru noul bloc
	initList(new_block, left_size, STATUS_FREE);
    // Actualizez referințele next și prev pentru noile blocuri
	new_block->next = block->next;
	new_block->prev = block;
	block->size = size;
	block->next = new_block;
    // Dacă există un bloc următor, actualizez prev pentru acel bloc
	if (new_block->next)
		new_block->next->prev = new_block;
}

struct block_meta *coalesce_blocks(void)
{
	struct block_meta *current = global_base;

	while (current && current->next) {
		struct block_meta *current_next = current->next;

		if (current->status == STATUS_FREE && current_next->status == STATUS_FREE) {
			// Unesc blocul curent și următorul bloc
			current->size += current_next->size;
			current->next = current_next->next;
			// Verific daca blocul urmator exista si fac legaturile
			if (current->next)
				current->next->prev = current;
		}	else {
			current = current->next;
		}
	}
	return current;
}

void *splitIfBigger(struct block_meta *block, int aligned_size)
{
	if  (block->size >= (size_t)(aligned_size + ALIGN(sizeof(struct block_meta) + ALIGN(1))))
		split_block(block, aligned_size);
	block->status = STATUS_ALLOC;
	return (void *)block + ALIGN(sizeof(struct block_meta));
}

void *malloc_calloc(size_t size, int isCalloc)
{
	if (size == 0)
		return NULL;
	int aligned_size = ALIGN(sizeof(struct block_meta)) + ALIGN(size);
    // Daca global base e NULL, prealloc memorie pt malloc sau calloc
	if (global_base == NULL && aligned_size < MMAP_THRESHOLD && isCalloc == 0) {
		global_base = sbrk(MMAP_THRESHOLD);
		DIE(global_base == MAP_FAILED, "sbrk");
		initList(global_base, MMAP_THRESHOLD, STATUS_FREE);
	}	else if (global_base == NULL && aligned_size < sysconf(_SC_PAGE_SIZE) && isCalloc == 1) {
		global_base = sbrk(MMAP_THRESHOLD);
		DIE(global_base == MAP_FAILED, "sbrk");
		initList(global_base, MMAP_THRESHOLD, STATUS_FREE);
	}
	struct block_meta *last = coalesce_blocks();
	struct block_meta *block = best_block_choice(aligned_size);

	if (block != NULL) {
		if (isCalloc == 1)
			memset((void *)block + ALIGN(sizeof(struct block_meta)), 0, size);
		return splitIfBigger(block, aligned_size);
	}	else if (last && last->status == STATUS_FREE) {
		block = (isCalloc == 0) ? request_space_malloc(aligned_size - last->size) :
		request_space_calloc(aligned_size - last->size);
		last->size += block->size;
		block = last;
		if (isCalloc == 1)
			memset((void *)block + ALIGN(sizeof(struct block_meta)), 0, size);
		return splitIfBigger(block, aligned_size);
	}
		block = (isCalloc == 0) ? request_space_malloc(aligned_size) : request_space_calloc(aligned_size);
		if (block->status == STATUS_ALLOC) {
			last->next = block;
			block->prev = last;
		}
		if (isCalloc == 1)
			memset((void *)block + ALIGN(sizeof(struct block_meta)), 0, size);
		return (void *)block + ALIGN(sizeof(struct block_meta));
}

void *os_malloc(size_t size)
{
	return malloc_calloc(size, 0);
}

struct block_meta *get_block_ptr(void *ptr)
{
	return (struct block_meta *)(ptr - ALIGN(sizeof(struct block_meta)));
}

void os_free(void *ptr)
{
	if (!ptr)
		return;
	struct block_meta *block = (struct block_meta *)(ptr - ALIGN(sizeof(struct block_meta)));

	if (block->status == STATUS_MAPPED) {
		// Daca blocul e alocat cu mmap, ii dau free
		void *ret = (void *)(size_t)munmap(block, block->size);

		DIE(ret == MAP_FAILED, "munmap");
	}	else {
		// Daca blocul e alocat cu sbkr, setez ca si free
		block->status = STATUS_FREE;
	}
}

void *os_calloc(size_t nmemb, size_t size)
{
	if (nmemb == 0)
		return NULL;
	size = size * nmemb;
	return malloc_calloc(size, 1);
}

int minim(int a, int b)
{
	if (a < b)
		return a;
	return b;
}

void *os_realloc(void *ptr, size_t size)
{
	if (!ptr)
		return os_malloc(size);
	if (size == 0) {
		os_free(ptr);
		return NULL;
	}
	struct block_meta *block = (struct block_meta *)(((void *)ptr) - ALIGN(sizeof(struct block_meta)));

	if (block->status == STATUS_FREE)
		return NULL;
	if (block->size == (size_t)ALIGN(size))
		return ptr;
	if (ALIGN(size) + ALIGN(sizeof(struct block_meta)) >= MMAP_THRESHOLD || block->status == STATUS_MAPPED) {
		void *new_ptr = os_malloc(size);

		memcpy(new_ptr, ptr, minim(block->size, ALIGN(size)));
		os_free(ptr);
		return new_ptr;
	}
	size_t aligned_size = ALIGN(size) + ALIGN(sizeof(struct block_meta));
    // Daca dimensiunea e mai mare, verific daca pot sa dau split
	if (block->size >= (size_t)ALIGN(size)) {
		if (block->size >= aligned_size + ALIGN(sizeof(struct block_meta)) + ALIGN(1))
			split_block(block, aligned_size);
		return ptr;
	}
	if (block->next == NULL) {
		void *ret = (void *)sbrk(ALIGN(size) - block->size);

		DIE(ret == MAP_FAILED, "Error sbrk");
		block->status = STATUS_ALLOC;
		block->size = ALIGN(size);
	}
	//Coalesce pt blocurile libere
	struct block_meta *next = block->next;

	while (next && next->status == STATUS_FREE && (size_t)ALIGN(size) > block->size) {
		block->size += next->size;
		block->next = next->next;
		//next->next->prev = block;
		next = block->next;
	}
	// Incerc sa extind blocul
	if (block->size >= aligned_size) {
		// Daca blocul e mai mare, ii dau split
		if (block->size >= aligned_size + ALIGN(sizeof(struct block_meta)) + ALIGN(1))
			split_block(block, aligned_size);
		return (void *)(((char *)block) + ALIGN(sizeof(struct block_meta)));
	}
    // Daca blocul nu poate fi extins, aloc un spatiu nou pt el
	void *new_ptr = os_malloc(size);

	memcpy(new_ptr, ptr, minim(block->size, ALIGN(size)));
	os_free(ptr);
	return new_ptr;
}
