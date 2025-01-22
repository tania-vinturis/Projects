
import os
import pygame

images = {}

def get_hangman_image(state: int):
    image_dir = os.path.join(os.path.dirname(__file__), 'images')
    
    image_path = os.path.join(image_dir, f"{state}.png")
    
    if state not in images:
        images[state] = pygame.image.load(image_path)
        
    return images[state]
