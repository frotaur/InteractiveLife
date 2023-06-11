import numpy as np
from numba import njit, prange,cuda 
import numba.typed as numt
import random
from PIL import Image, ImageEnhance


class Automaton :
    """
        Class that internalizes the rules and evolution of 
        the cellular automaton at hand. It has a step function
        that makes one timestep of the evolution. By convention,
        and to keep in sync with pygame, the world tensor has shape
        (W,H,3). It contains float values between 0 and 1, which
        are mapped to 0 255 when returning output, and describes how the
        world is 'seen' by an observer.

        Parameters :
        size : 2-uple
            Shape of the CA world
        
    """

    def __init__(self,size):
        self.w, self.h  = size
        self.size= size
        self._worldmap = np.random.uniform(size=(self.w,self.h,3))
    

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    @property
    def worldmap(self):
        return (255*self._worldmap).astype(dtype=np.uint8)



class SMCA(Automaton):
    """
        Standard Model Cellular Automaton. Inspired by LGCA.

        Parameters :
        <put them as I go>
    """
    def convertxy(self,rule : tuple[str]):
        """
            Converts easy to understand rule in two ints, which
            are easier to handle.
        """
        rulef =[0,0]
        for i,rulestr in enumerate(rule):
            if(rulestr==""):
                rulef[i]=0
                break

            for number in rulestr:
                rulef[i]+=2**(int(number))
        
        return tuple(rulef)
    
    def __init__(self, size, rule :tuple[str]):
        super().__init__(size)
        # 0,1,2,3 are  N,O,S,E directions
        self.particles = np.random.randn(1,self.w,self.h) # (1,W,H)
        self.particles = np.where(self.particles>1.9,1,0).astype(np.int16)
        
        self.rule=self.convertxy(rule)

        self.dir = np.array([[0,-1],[-1,0],[0,1],[1,0]])

        self.emission_p = 1.
        self.interaction_p = 1.

        self.brush_size=3

        self.limmin = np.array([0,0])
        self.limmax = np.array([self.w,self.h])
    def evolve_step(self):
        self.particles=evolve_cpu(self.particles,self.w,self.h,self.rule[0],self.rule[1])
        
                    
    def step(self):
        self.evolve_step()
    
    def update_map(self):
        self._worldmap = np.zeros_like(self._worldmap)
        self._worldmap[:,:,:]+=self.particles[0,:][:,:,None]
    
    def clamp_coord(self,v):
        return np.array([min(max(0,v[0]),self.w),min(max(0,v[1]),self.h)])

    def add_partic(self,x,y,random=False):
        v = np.array([x,y])
        vmin = self.clamp_coord(v-self.brush_size+1)
        vmax = self.clamp_coord(v+self.brush_size)
        if(random):
            self.particles[0,vmin[0]:vmax[0],vmin[1]:vmax[1]]=np.random.random_integers(0,1,tuple(vmax-vmin))
        else :
            self.particles[0,vmin[0]:vmax[0],vmin[1]:vmax[1]]= 1

    def load_state(self,state):
        """
            Load the state of the world. 

            params : 
            state : tensor (1,W,H) of 0 or 1.
        """

        self.particles=state

    def state_from_picture(self,pic_loc):
        """
            Loads state from picture.
        """
        img = Image.open(pic_loc)
        img = ImageEnhance.Brightness(img).enhance(0.5)
        img = img.convert('L')
        img.thumbnail((self.w,self.h))

        img = (np.array(img).transpose(1,0))[None,:,:]
        # Calculate padding
        print(f"h,w : {self.h,self.w}")
        pad_height = max(self.h - img.shape[2],0)
        pad_width = max(self.w - img.shape[1],0)
        print(f"pad_height :{pad_height}")
        print(f"pad_width : {pad_width}")
        # Ensure padding is equally split between top/bottom and left/right
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        print(f'Before shape : {img.shape}')
        print(f'Pad top : {pad_top}')
        print(f'Pad Bottom : {pad_bottom}')
        print(f'pad left : {pad_left}')
        print(f'pad right : {pad_right}')

        img = np.pad(img, ((0,0), (pad_left, pad_right),(pad_top, pad_bottom)), mode='constant', constant_values=0)
        print('AFTER SHAPE : ', img.shape)
        assert img.shape==(1,self.w,self.h)

        self.particles = np.where(img>55,1,0)

    def remove_partic(self,x,y):
        v = np.array([x,y])
        vmin = self.clamp_coord(v-self.brush_size+1)
        vmax = self.clamp_coord(v+self.brush_size)

        self.particles[0,vmin[0]:vmax[0],vmin[1]:vmax[1]]=0
  

@njit(parallel=True)
def evolve_cpu(partics,w,h,rulex,ruley) :
    newpartics=np.copy(partics)
    for x in prange(w):
        for y in prange(h):
            sum=0
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if(i!=0 or j!=0):
                        sum+=partics[0,(x+i)%w,(y+j)%h]
            if(partics[0,x,y]==1):
                if(not((rulex >> sum) & 1)):
                    newpartics[0,x,y]=0
            if(partics[0,x,y]==0):
                if((ruley >> sum) & 1):
                    newpartics[0,x,y]=1

    return newpartics


# NICE BUT WRONG
# @njit(parallel=True)
# def evolve_cpu(partics,w,h,rulex,ruley) :
#     newpartics=np.copy(partics)
#     for x in prange(w):
#         for y in prange(h):
#             sum=0
#             for i in [-1,0,1]:
#                 for j in [-1,0,1]:
#                     if(i!=0 or j!=0):
#                         sum+=partics[0,(x+i)%w,(y+j)%h]
#             if(partics[0,x,y]==1):
#                 if((rulex >> sum) & 1):
#                     newpartics[0,x,y]=1
#             if(partics[0,x,y]==0):
#                 if((ruley >> sum) & 1):
#                     newpartics[0,x,y]=1

#     return newpartics