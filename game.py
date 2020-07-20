import pygame
import neat
import time
import os
import random

win_width = 500
win_height = 800
pygame.font.init()
font = pygame.font.SysFont("comicsans",50)
birds = [pygame.transform.scale2x(pygame.image.load(os.path.join("images","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("images","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("images","bird3.png")))]
pipe = pygame.transform.scale2x(pygame.image.load(os.path.join("images","pipe.png")))
base = pygame.transform.scale2x(pygame.image.load(os.path.join("images","base.png")))
background = pygame.transform.scale2x(pygame.image.load(os.path.join("images","bg.png")))
generation = 0
DRAW_LINES = True
#vel = velocity

class Pipe:
    gap = 200
    vel = 5
    def __init__(self,x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.pipe_top = pygame.transform.flip(pipe,False,True)
        self.pipe_bottom = pipe
        self.passed = False
        self.set_height()
    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.pipe_top.get_height()
        self.bottom = self.height+self.gap
    def move(self):
        self.x -= self.vel
    def draw(self,window):
        window.blit(self.pipe_top,(self.x,self.top))
        window.blit(self.pipe_bottom,(self.x,self.bottom))
    def collide(self,bird):
        bird_mask = bird.get_mask()
        top_pipe_mask =  pygame.mask.from_surface(self.pipe_top)
        bottom_pipe_mask = pygame.mask.from_surface(self.pipe_bottom)
        toffset = (self.x-bird.x,self.top-round(bird.y))
        boffset = (self.x-bird.x,self.bottom-round(bird.y))
        bpoint = bird_mask.overlap(bottom_pipe_mask,boffset)#bpoint is none if it does not collide
        tpoint = bird_mask.overlap(top_pipe_mask,toffset)
        if tpoint or bpoint:
            return True
        return False
class Base:
    vel = 5
    width = base.get_width()
    img = base
    def __init__(self,y):
        self.y= y
        self.x1 = 0
        self.x2 = self.width
    def move(self):
        self.x1-=self.vel
        self.x2-=self.vel
        if self.x1+self.width<0:
            self.x1 = self.x2+self.width
        if self.x2+self.width<0:
            self.x2 = self.x1+self.width
    def draw(self,window):
        window.blit(self.img,(self.x1,self.y))
        window.blit(self.img,(self.x2,self.y))
class Bird:
    IMGS = birds
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_COUNT = 5
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0#velocity
        self.height = self.y
        self.img_count = 0
        self.image = self.IMGS[0]
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    def move(self):
        self.tick_count+=1#this is like time
        displacement = self.vel*self.tick_count+(0.5)*3*self.tick_count**2#s = ut+1/2at^2 accelaration-3
        if displacement>=16:
            displacement = 16
        if displacement<0:
            displacement -=2
        self.y = self.y+displacement
        if displacement<0 or self.y < self.height +50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL
    def draw(self,window):
        self.img_count += 1
        if self.img_count< self.ANIMATION_COUNT:
            self.image = self.IMGS[0]
        elif self.img_count< self.ANIMATION_COUNT*2:
            self.image = self.IMGS[1]
        elif self.img_count< self.ANIMATION_COUNT*3:
            self.image = self.IMGS[2]
        elif self.img_count == self.ANIMATION_COUNT*4+1:
            self.image = self.IMGS[0]
            self.img_count = 0
        if self.tilt <=-80:
            self.image = self.IMGS[1]
        #rotating an image in pygame
        rotated_image = pygame.transform.rotate(self.image,self.tilt)
        new_rect = rotated_image.get_rect(center=self.image.get_rect(topleft = (self.x,self.y)).center)
        window.blit(rotated_image,new_rect.topleft)
    def get_mask(self):
        return pygame.mask.from_surface(self.image)

def draw_window(window,birds,pipes,base,score,gen,pipe_ind):
    window.blit(background,(0,0))
    if gen==0:
        gen=1
    for pipe in pipes:
        pipe.draw(window)
    text = font.render("Score - "+str(score),1,(255,255,0))
    window.blit(text,(win_width-10-text.get_width(),10))

    generations =  font.render("Generaton - "+str(gen-1),1,(255,0,255))
    window.blit(generations,(win_width-10-generations.get_width(),win_height-100))

    bird_score =  font.render("Birds Alive  - "+str(len(birds)),1,(255,0,255))
    window.blit(bird_score,(win_width-10-bird_score.get_width(),win_height-150))

    base.draw(window)
    for bird in birds:
        if DRAW_LINES:
            try:
                pygame.draw.line(window, (255,0,0), (bird.x+bird.image.get_width()/2, bird.y + bird.image.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].pipe_top.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(window, (255,0,0), (bird.x+bird.image.get_width()/2, bird.y + bird.image.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].pipe_bottom.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        bird.draw(window)
    pygame.display.update()

def main(genomes,config):
    nets = []
    gen = []
    birds = []
    global generation
    generation +=1
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        gen.append(g)

    base = Base(730)
    pipes = [Pipe(600)]
    run = True
    score = 0
    clc = pygame.time.Clock()
    window = pygame.display.set_mode((win_width,win_height))
    while run:
        clc.tick(30)
        for evnt in pygame.event.get():
            if evnt.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].pipe_top.get_width():
                pipe_ind = 1
        else:
            run =False
            break
        for x,bird in enumerate(birds):
            bird.move()
            gen[x].fitness+=0.1
            output = nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].bottom)))
            if output[0]>0.5:
                bird.jump()
        remove = []
        add_pipe = False
        for pipe in pipes:
            for x,bird in enumerate(birds):
                if pipe.collide(bird):
                    gen[x].fitness -=1
                    birds.pop(x)
                    nets.pop(x)
                    gen.pop(x)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            if pipe.x+pipe.pipe_top.get_width()<-1:
                    remove.append(pipe)
            pipe.move()
        if add_pipe:
            score+=1
            for g in gen:
                g.fitness +=5#if a bird pass then give thema reward
            pipes.append(Pipe(600))
        for r in remove:
            pipes.remove(r)
        for x,bird in enumerate(birds):
            if bird.y + bird.image.get_height() >=730 or bird.y<0:
                birds.pop(x)
                nets.pop(x)
                gen.pop(x)
        base.move()
        draw_window(window,birds,pipes,base,score,generation,pipe_ind)


def run(Config):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         Config)
    p = neat.Population(config)#setting up the config file

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main,50)#setting the fitness function
    print('\nBest genome:\n{!s}'.format(winner))

if __name__=="__main__":
    loc_dir = os.path.dirname(__file__)
    generation = 0
    config = os.path.join(loc_dir, "CONFIG.txt")
    run(config)
