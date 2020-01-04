import pygame
import time
import random
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from DQN import Agent

agent = Agent()
noOfGames = 0
record = 0
GAMESIZE = 405
blocksize = 15
blocks = GAMESIZE//blocksize

def getRecord(score, record):
    if score>record:
        return score
    return record

def collision(pos1, pos2, bsize):
    if pos1.x == pos2.x and pos1.y == pos2.y:
        return True
    return False

def outOfBounds(pos):
    if pos.x < 0 or pos.x > blocks-2 or pos.y < 0 or pos.y > blocks-2:
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
            
    def doMove(self, move):
        if np.array_equal(move ,[1, 0, 0]):
            return
        if np.array_equal(move ,[0, 1, 0]):
            if self.dir == 1:
                self.dir = 4
            else:
                self.dir -= 1
        if np.array_equal(move ,[0, 0, 1]):
            if self.dir == 4:
                self.dir = 1
            else:
                self.dir += 1
    
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
        self.pos = pygame.Vector2(self.pos.x + self.mov.x, self.pos.y + self.mov.y)
        
    def draw(self, gameDisplay):
        pygame.draw.rect(gameDisplay, (0,255,0), (self.pos.x*self.blocksize,self.pos.y*self.blocksize,self.blocksize,self.blocksize))
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

    windowWidth = 405
    windowHeight = 405


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
        self.state = [self.screenshot(), self.screenshot()]
        self.next_state = [self.screenshot(), self.screenshot()]
    
    def screenshot(self):
        data = pygame.image.tostring(self._display_surf, 'RGB')  # Take screenshot
        image = Image.frombytes('RGB', (self.windowWidth, self.windowWidth), data)
        image = image.convert('L')  # Convert to greyscale
        image = image.resize((84, 84))  # Resize
        image = image.convert('1')
        matrix = np.asarray(image.getdata(), dtype=np.float64)
        return matrix.reshape(image.size[0], image.size[1])

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
                print("Game number: "+str(noOfGames+1)+" Score: "+str(self.snake.length-4))
                self._running = False
                break
            
        pass
    
    def on_render(self):
        self._display_surf.fill((0,0,0))
        self.snake.draw(self._display_surf)
        self.food.draw(self._display_surf)
        scoretext =  pygame.font.SysFont("monospace", 15).render("Score = "+str(self.snake.length-4), 1, (255,255,255))
        recordtext =  pygame.font.SysFont("monospace", 15).render("Record = "+str(record), 1, (255,255,255))
        self._display_surf.blit(scoretext, (5, 10))
        self._display_surf.blit(recordtext, (5, 25))
        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        global record
        if self.on_init() == False:
            self._running = False
        
 
        while( self._running ):
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                if (keys[pygame.K_ESCAPE]):
                    print(self.screenshot())
                    self._running = False
                    pygame.quit()
                    quit()
            agent.epsilon = 45 - noOfGames//4
            
            state_old = agent.getState(self)
            
            if random.randint(0,100) < agent.epsilon:
                move = to_categorical(random.randint(0,2), num_classes = 3)
            else:
                prediction = agent.model.predict(state_old.reshape((1,11)))
                move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
            
            self.snake.doMove(move)
            self.on_loop()
            
            state_new = agent.getState(self)
            reward = agent.setReward(self)
            
            agent.shortMemory(state_old, move, reward, state_new, self._running)
            agent.remember(state_old, move, reward, state_new, self._running)
            record = getRecord(self.snake.length-4, record)
            
            self.on_render()

            time.sleep (50.0 / 1000.0);
        self.on_cleanup()


if __name__ == "__main__" :
    while noOfGames < 200:
        game = App()   
        game.on_execute()
        agent.replay(agent.memory)
        print(agent.model.weights)
        noOfGames += 1
    agent.model.save_weights('weights.hdf5')
