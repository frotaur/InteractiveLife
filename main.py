import pygame
from Camera import Camera
from Automaton import *
import cv2 



def make_recording(frames,writer):
    frames = frames
    for frame in frames :
        writer.write(frame)
    print(f'there are {len(frames)} frames')
    writer.release()
    print('DONE !')


pygame.init()
W,H =64,64
screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)

world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

rules = {'chaosblob':("56","24"), 'life': ("23","3"), 'seeds':("","2"),
         'free':("0","2"),'explolife':("13","2"),'serviette':("","234"),
         'maze':("1234","3"),'pulsar':("238","3"),'dead':("12","3")}
auto = SMCA((W,H),rules['life'])
auto.fade_speed=8.
stopped=True
add_drag = False
rem_drag = False
recording=False

auto.state_from_picture('galaxy.jpg')


fourcc =  cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a codec that works with .mp4 files. For .avi files, use 'XVID'
video = cv2.VideoWriter('life.mp4', fourcc, 24, (W, H))

frames=[]

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        camera.handle_event(event)
        if event.type == pygame.MOUSEBUTTONDOWN and not(pygame.key.get_mods() & pygame.KMOD_META) :
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
                print(x,y)
                auto.add_partic(int(x-1),int(y),random=True)
            elif(rem_drag):
                x,y=pygame.mouse.get_pos()
                auto.remove_partic(int(x-1),int(y))
    
        if event.type == pygame.KEYDOWN :
            if(event.key == pygame.K_SPACE):
                stopped=not(stopped)
            if(event.key == pygame.K_r):
                recording = not recording
                print('REC')
            if(event.key == pygame.K_BACKSPACE):
                auto.reset()
            if(event.key == pygame.K_p):
                auto.change_brush_size(1)
            if(event.key == pygame.K_m):
                auto.change_brush_size(-1)
            if(event.key == pygame.K_s):
                auto.step()
            

    if(not stopped):
        auto.step()
    auto.update_map()
    world_state = auto.worldmap
    # world_state = np.zeros_like(auto.worldmap)
    # world_state[np.random.randint(0,W,(10000,)),np.random.randint(0,H,(10000,)),:]=255
    surface = pygame.surfarray.make_surface(world_state)

    if(recording):
         frame = world_state.transpose(1,0,2)
         tempB = np.copy(frame[:,:,2])
         frame[:,:,2]=frame[:,:,0]
         frame[:,:,0]=tempB
         frames.append(frame)
    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(8)  # limits FPS to 60


pygame.quit()

if(len(frames)>0):
    make_recording(frames,video)