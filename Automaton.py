import numpy as np
from numba import njit, prange 
import random
from PIL import Image, ImageEnhance
import torch
from torchenhanced import DevModule
from torchenhanced.util import showTens

class Automaton(DevModule) :
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

    def __init__(self,size,device='cpu'):
        super().__init__(device)
        self.h, self.w  = size
        self.size= size
        self._worldmap = np.zeros((self.w,self.h,3))

        self.to(device)

    def step(self):
        return NotImplementedError('Please subclass "Automaton" class, and define self.step')
    
    def draw(self):
        # Should compute _worldmap given state
        pass
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
        self.particles = np.where(self.particles>1.9,1,0).astype(np.int)

        self.rule=self.convertxy(rule)



        self.brush_size=3

        self.limmin = np.array([0,0])
        self.limmax = np.array([self.w,self.h])

        self.background = np.array([35.,35.,51.])/255.

    def evolve_step(self):
        self.particles=evolve_cpu(self.particles,self.w,self.h,self.rule[0],self.rule[1])
        
                    
    def step(self):
        self.evolve_step()
    
    def update_map(self):
        self._worldmap[:,:,1] -= 0.1
        self._worldmap[:,:,2] -= 0.2
        self._worldmap[:,:,0] -= 0.05
        self._worldmap[self._worldmap<0] =0.
        
        self._worldmap[:,:,:]+=self.particles[0,:][:,:,None]
        self._worldmap = np.minimum(self._worldmap,1)
        self._worldmap = np.maximum(self._worldmap,self.background[None,None,:])
    
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


class LifeLikeCA(Automaton) :
    def __init__(self,size, rule :tuple[str], device='cpu'):
        super().__init__(size)
        self.h,self.w = size
        self.size = torch.tensor(size)
        self.x,self.y = self.to_unreadable([rule[0]],[rule[1]])

        self.state = self.get_init_mat_varied(batch_size=1,portion_range=(0.5,1.)) # (1,H,W)

        self.background = np.array([35.,35.,51.])/255.

        print('the world state is : ', self.state.shape)
    
    def set_rule(self,rule):
        self.x,self.y = self.to_unreadable([rule[0]],[rule[1]])
    def get_init_mat(self,rand_portion,flip_bw=False,batch_size=1):
        """
            Get initialization matrix for CA

            Params : 
            rand_portion : tensor, portion of the screen filled with noise.
            flip_bw : bool, flips black and white.
            batch_size : int, number of initial conditions to generate.
        """
        batched_size = torch.tensor([batch_size,self.h,self.w])
        randsize = (batched_size[1:]*rand_portion).to(dtype=torch.int) # size of the random square
        randstarts = (batched_size[1:]*(1-rand_portion)/2).to(dtype=torch.int) # Where to start the index for the random square

        randsquare = torch.where(torch.randn((batch_size,*randsize.tolist()))>0,1,0) # Creates random square of 0s and 1s

        init_mat = torch.zeros((batch_size,self.h,self.w),dtype=torch.int)
        init_mat[:,randstarts[0]:randstarts[0]+randsize[0],
        randstarts[1]:randstarts[1]+randsize[1]] = randsquare
        init_mat = init_mat.to(torch.int)

        if(flip_bw):
            init_mat=1-init_mat

        return init_mat.to(device=self.device, dtype=torch.int) # (B,H,W)

    def get_init_mat_varied(self,batch_size=1,portion_range:tuple=(0.5,1.)):
        """
             initalization matrix (B,H,W), where each example is varied in terms
             of the portion of the screen filled with random noise. Also flips b/w
             randomly.

             Args :
             batch_size : self-explanatory
             portion_range : range in which to generate the portions of noise for the inital condition
        """
            # Convert portions to a tensor
        portions_tensor = torch.rand((batch_size,))*(portion_range[1]-portion_range[0])+portion_range[0]

        # Calculate the size of the square for the entire batch
        square_H = (self.h * portions_tensor).int()
        square_W = (self.w * portions_tensor).int()

        # Calculate starting and ending indices for the height and width for the entire batch
        start_H = ((self.h - square_H) // 2).int()
        end_H = (start_H + square_H).int()

        start_W = ((self.w - square_W) // 2).int()
        end_W = (start_W + square_W).int()

        # Create a coordinate grid for height and width
        rows = torch.arange(self.h).view(1, self.h, 1).expand(batch_size, -1, self.w)
        cols = torch.arange(self.w).view(1, 1, self.w).expand(batch_size, self.h, -1)

        # Create the masks based on computed indices
        mask = ((rows >= start_H[:,None,None]) & (rows < end_H[:,None,None]) & (cols >= start_W[:,None,None]) & (cols < end_W[:,None,None])).float() # 1 if in correct portion

        init_mat = (torch.randint(0,2,size=(batch_size,self.h,self.w))*mask).int()

        flip_bw = (torch.randn(size=(batch_size,))>0)[:,None,None].expand(-1,self.h,self.w)

        init_mat = torch.where(flip_bw,init_mat,1-init_mat)

        return init_mat.to(device=self.device, dtype=torch.int) # (B,H,W)
    
    def evo_step(self,mat,x:int,y:int):
        """
            Evolves one step, using the 'un-readable' convention for rules x and y. In binary, the presence of 1 in the location d of x(y) means
            that if a live(dead) cell has d live neighbors, then it will survive(be born).

            params :
            mat : (1,H,W) matrix of state
            x,y : ints, rules for survival and births respectively.
        """
        mat = mat[0]
        wmat, emat = mat.roll(-1, 0), mat.roll(1, 0) # the second argument is the roll axis
        nmat, smat = mat.roll(-1, 1), mat.roll(1, 1) 
        swmat, semat = wmat.roll(1, 1), emat.roll(1, 1)
        nwmat, nemat = wmat.roll(-1, 1), emat.roll(-1, 1)

        count_mat = wmat + emat + nmat + smat + swmat + semat + nwmat + nemat

        return torch.where(mat==1,self.get_nth_bit(x,count_mat),self.get_nth_bit(y,count_mat)).to(torch.int)[None]

    def step(self):
        self.state = self.evo_step(self.state,self.x,self.y)

    def to_unreadable(self,x : list[str],y : list[str]):
        """
            Transform classic notation for x/y (see wikipedia article), to the more convenient notation
            that I designed, explained in evo_step. EX : S23/B3 is game of life, translated to S12/B8 in 
            my notation.

            <MAYBE LATER CHANGE TO TREAT RULES AS TUPLES (X,Y) INSTEAD OF SEPARATELY>
            params :
            x,y : list of rules in classic notation.

            returns :
            x,y : tensor of rules in my notation.            
        """
        x_out = []
        y_out = []

        # Iterate over x and y
        for rule in x:
            if(rule==''):
                x_out.append(0)
            else:
                x_out.append(sum([2**int(d) for d in rule]))
        
        for rule in y:
            if(rule==''):
                y_out.append(0)
            else:
                y_out.append(sum([2**int(d) for d in rule]))
        
        print('got rules : ', x_out, y_out)
        return torch.tensor(x_out,dtype=torch.int),torch.tensor(y_out,dtype=torch.int)

    def get_nth_bit(self,num, n):
        """
        Returns the nth bit (counting from the right, starting at 0) of num in binary representation.
            n can be a tensor of integers. Num can ALSO be a tensor
        """

        return (num>>n) & 1

    def draw(self):
        """
            Updates the worldmap with the current state.
        """
        self._worldmap[:,:,1] -= 0.07
        self._worldmap[:,:,2] -= 0.15
        self._worldmap[:,:,0] -= 0.01
        self._worldmap[self._worldmap<0] =0.

        self._worldmap[:,:,:]+=self.state[0].transpose(1,0)[:,:,None].cpu().numpy()
        self._worldmap = np.minimum(self._worldmap,1)
        self._worldmap = np.maximum(self._worldmap,self.background[None,None,:])

    def state_from_picture(self,pic_loc):
        """
            Loads state from picture.
        """
        img = Image.open(pic_loc)
        img = ImageEnhance.Brightness(img).enhance(0.5)
        img = img.convert('L')
        img.thumbnail((self.w,self.h))

        img = (np.array(img))[None,:,:] # (1,H,W)
        print('img shape : after transpo : ',img.shape)
        # Calculate padding
        pad_height = max(self.h - img.shape[1],0)
        pad_width = max(self.w - img.shape[2],0)

        # Ensure padding is equally split between top/bottom and left/right
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left


        img = np.pad(img, ((0,0), (pad_top, pad_bottom),(pad_left, pad_right)), mode='constant', constant_values=0)
        assert img.shape==(1,self.h,self.w), f"img shape : {img.shape}, expected : {(1,self.h,self.w)}"

        self.state = torch.tensor(np.where(img>55,1,0),device=self.device).to(torch.int)
