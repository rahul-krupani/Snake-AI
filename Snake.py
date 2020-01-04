import pygame
import time
import random

def collision(pos1, pos2, bsize):
#    if pos1.x >= pos2.x and pos1.x <= pos2.x + bsize:
#        if pos1.y >= pos2.y and pos1.y <= pos2.y + bsize:
    if pos1.x == pos2.x and pos1.y == pos2.y:
        return True
    return False

def outOfBounds(pos):
    if pos.x < 0 or pos.x > 38 or pos.y < 0 or pos.y > 38:
        return True
    return False

class Food:
    pos = pygame.Vector2(0,0);
    x = 0
    y = 0
    blocksize = 15
    def __init__(self, game_width, game_height):
        self.blocks = (game_width//self.blocksize)-3
        self.pos.x = random.randint(0,self.blocks)
        self.pos.y = random.randint(0,self.blocks)
    
    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, (255,0,0), (self.pos.x*self.blocksize, self.pos.y*self.blocksize, self.blocksize, self.blocksize))

class Snake:
    def __init__(self, game_width, game_height):
        self.game_width = game_width
        self.game_height = game_height
        self.blocksize = 15
        self.pos = pygame.Vector2(game_width//(2*self.blocksize), game_height//(2*self.blocksize))
        self.length = 4
        self.positions = []
        self.mov = pygame.Vector2(1, 0)
        self.dir = 1
        self.eaten = False
        for i in range(3):
            self.positions.append(pygame.Vector2(self.pos.x, self.pos.y + (i+1)))
    
    def changeDir(self):
        if self.dir==1:
            self.mov = pygame.Vector2(1, 0)
        elif self.dir==2:
            self.mov = pygame.Vector2(0, -1)
        elif self.dir==3:
            self.mov = pygame.Vector2(-1, 0)
        else:
            self.mov = pygame.Vector2(0, 1)
    
    def move(self):
        self.positions.insert(0, self.pos)
        #del self.positions[self.length-1]
        self.pos = pygame.Vector2(self.pos.x + self.mov.x, self.pos.y + self.mov.y)
        
    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, (255,255,255), (self.pos.x*self.blocksize,self.pos.y*self.blocksize,self.blocksize,self.blocksize))
        for i in range(self.length-1):
            pygame.draw.rect(gameDisplay, (255,255,255), (self.positions[i].x*self.blocksize,self.positions[i].y*self.blocksize,self.blocksize,self.blocksize))

    def update(self, gameDisplay):
        if self.eaten:
            self.length+=1
            self.eaten=False
        self.changeDir()
        self.move()
        self.draw(gameDisplay)

class App:

    windowWidth = 600
    windowHeight = 600


    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.snake = Snake(self.windowWidth, self.windowHeight) 
        self.food = Food(self.windowWidth, self.windowHeight)

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
        
        pygame.display.set_caption('Snake')
        self._running = True
        self.snake.draw(self._display_surf)
        self.food.draw(self._display_surf)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        self.snake.update(self._display_surf)
        self.food.draw(self._display_surf)
        
        for i in range(self.snake.length-1):
            if collision(self.food.pos, self.snake.positions[i], 15):
                self.snake.eaten = True
                self.food.pos = pygame.Vector2(random.randint(0,self.food.blocks),random.randint(0,self.food.blocks))
                self.food.draw(self._display_surf)
        
        for i in range(self.snake.length-1):
            if collision(self.snake.pos, self.snake.positions[i], 15) or outOfBounds(self.snake.pos):
                print("You lose! Score: "+str(self.snake.length-4))
                self._running = False
                break
            
        pass
    
    def on_render(self):
        self._display_surf.fill((0,0,0))
        self.snake.draw(self._display_surf)
        self.food.draw(self._display_surf)
        scoretext =  pygame.font.SysFont("monospace", 20).render("Score = "+str(self.snake.length-4), 1, (255,255,255))
        self._display_surf.blit(scoretext, (5, 10))
        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                keys = pygame.key.get_pressed() 
            
                if (keys[pygame.K_RIGHT]):
                    if not(self.snake.dir == 1 or self.snake.dir == 3):
                        self.snake.dir = 1
    
                if (keys[pygame.K_LEFT]):
                    if not(self.snake.dir == 1 or self.snake.dir == 3):
                        self.snake.dir = 3
    
                if (keys[pygame.K_UP]):
                    if not(self.snake.dir == 2 or self.snake.dir == 4):
                        self.snake.dir = 2
    
                if (keys[pygame.K_DOWN]):
                    if not(self.snake.dir == 2 or self.snake.dir == 4):
                        self.snake.dir = 4
    
                if (keys[pygame.K_ESCAPE]):
                    self._running = False

            self.on_loop()
            self.on_render()

            time.sleep (50.0 / 1000.0);
        self.on_cleanup()


if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()