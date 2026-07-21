import pygame
from Camera import Camera
from Automaton import *
import cv2, os



def make_recording(frames,writer):
    if(invert_video):
        frames = frames[::-1]
    for frame in frames :
        writer.write(frame)
    writer.release()
    
    print('DONE !')


def make_writer():
    global video, fourcc
    video_dir = 'videos'
    os.makedirs(video_dir, exist_ok=True)
    numvids = len(os.listdir(video_dir))
    video = cv2.VideoWriter(os.path.join(video_dir,f'{numvids}.mp4'), fourcc, 60, (W, H))

pygame.init()
# NOT full-HD display
H,W = 460,640
invert_video = True
picture = 'titlescreen.png'




screen = pygame.display.set_mode((W,H),flags=pygame.SCALED|pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True
camera = Camera(W,H)



world_state = np.random.randint(0,255,(W,H,3),dtype=np.uint8)

rules = {'chaosblob':("5678","25678"), 'life': ("23","3"), 'gnarl':("1","1"), 'replicator':("1357","1357"),
         'maze':("12345","3"),"coral":("45678","3"),"coagulations":("235678","378"),"walled_cities":("45678","45678"),
        "amoeba":("1358","357"),"assimilation":("4567","345"),"diamoeba":("5678","3567"),"day_and_night":("34678","3678"),
        "highlife":("23","36"),"stains":("235678","3678"),"2x2":("36","125"),"34":("34","34"),"longlife":("5","345"),"move":("245","368"),
        "pseudo_life":("357","238"),"serviettes":("234","3"),"stains":("235678","3678"),"vote":("5678","45678"),"walled_maze":("1234","3"),
        "replicator":("1357","1357"),"2x2":("36","125"),"34":("34","34"),"amoeba":("1358","357"),"assimilation":("4567","345"),
        "coagulations":("235678","378"),"coral":("45678","3"),"day_and_night":("34678","3678"),"diamoeba":("5678","3567"),
        "flakes":("012345678","3"),"gnarl":("1","1"),"highlife":("23","36"),"life":("23","3"),"longlife":("5","345"),
        "maze":("12345","3"),"move":("245","368"),"pseudo_life":("357","238"),"serviettes":("234","3"),"stains":("235678","3678"),
        "vote":("5678","45678"),"walled_cities":("45678","45678"),"walled_maze":("1234","3"),"ringsandslugs":("14568","56"),
         "vote45":("35678","4678") }

newrules = {'feux':("0247","1358"), 'snakeskin':("134567","1")}
rulelist = list(rules.values())
rulenum = 0

picturelist= [os.path.join('oldscreens',pic) for pic in os.listdir('oldscreens')]
picturenum = 0
auto = LifeLikeCA((H,W),rules['life'],device='cpu')

stopped=True


add_drag = False
rem_drag = False
recording=False

auto.state_from_picture('pic.jpg')


fourcc =  cv2.VideoWriter_fourcc(*'H264')  # 'mp4v' is a codec that works with .mp4 files. For .avi files, use 'XVID'
video = None

make_writer()

frames=[]

fuckface = [(472, 24), (355,401), (496,346), (447,148), (198,360)]
fucki = 0
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
            if(event.key == pygame.K_r):
                if(recording):
                    make_recording(frames,video)
                    frames=[]
                    recording=False
                    make_writer()
                    print('DONE !')
                else:
                    recording = True
                print('REC')
            if(event.key == pygame.K_i):
                auto.state_from_picture(picture)
            if(event.key == pygame.K_s):
                auto.set_rule(rulelist[rulenum])
                rulenum = (rulenum+1)%len(rulelist)
            if(event.key == pygame.K_n):
                picture = picturelist[picturenum]
                picturenum = (picturenum+1)%len(picturelist)
                auto.state_from_picture(picture)
            if(event.key == pygame.K_d):
                # auto.x = torch.tensor(np.random.randint(0,512,(1,)),dtype=torch.int,device=auto.device)
                # auto.y = torch.tensor(np.random.randint(0,512,(1,)),dtype=torch.int,device=auto.device)
                fucker = fuckface[fucki]
                auto.x = torch.tensor([fucker[0]])
                auto.y = torch.tensor([fucker[1]])
                fucki+=1
                print('rule : ', auto.x.cpu().item(), auto.y.cpu().item())

    if(not stopped):
        auto.step()
    auto.draw()
    world_state = auto.worldmap
    surface = pygame.surfarray.make_surface(world_state)

    if(recording):
        frame = world_state.transpose(1,0,2)
        tempB = np.copy(frame[:,:,2])
        frame[:,:,2]=frame[:,:,0]
        frame[:,:,0]=tempB
        frames.append(world_state.transpose(1,0,2))

        # Draw a red rectangle on the bottom right
        pygame.draw.rect(surface, (255, 0, 0), (W-50, H-50, 10, 10))
    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the scaled surface on the window
    zoomed_surface = camera.apply(surface)

    screen.blit(zoomed_surface, (0,0))

    # Update the screen
    pygame.display.flip()
    # flip() the display to put your work on screen

    clock.tick(60)  # limits FPS to 60


pygame.quit()

if(len(frames)>0):
    make_recording(frames,video)