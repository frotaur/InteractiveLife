import pygame
from Camera import Camera
from Automaton import *

pygame.init()
W,H =800,600
screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

auto = SMCA((W,H),("23","3"))
stopped=True
add_drag = False
rem_drag = False

auto.state_from_picture('name.png')
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN :
            if(event.button == 1):#left click
                add_drag=True
            if(event.button ==3):
                rem_drag=True
        if event.type == pygame.MOUSEBUTTONUP:
            if(event.button==1):
                add_drag=False
            elif(event.button==3):
                rem_drag=False
        if event.type == pygame.MOUSEMOTION:
            if(add_drag):
                x,y=pygame.mouse.get_pos()
                auto.add_partic(x,y)
            elif(rem_drag):
                x,y=pygame.mouse.get_pos()
                auto.remove_partic(x,y)
    
        if event.type == pygame.KEYDOWN :
            if(event.key == pygame.K_SPACE):
                stopped=not(stopped)
        
    
    if(not stopped):
        auto.step()
    auto.update_map()
    world_state = auto.worldmap
    surface = pygame.surfarray.make_surface(world_state)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(24)  # limits FPS to 60

pygame.quit()

