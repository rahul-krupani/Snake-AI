from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import pygame
import time
import random
import numpy as np
from keras.utils import to_categorical
from collections import deque
import json
    
noOfGames = 0
record = 0
GAMESIZE = 330
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

####MODEL####
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4

img_rows , img_cols = 84, 84

def buildmodel():
    model = Sequential()
    model.add(Dense(activation="relu", input_dim=11, units=120))
    model.add(Dropout(0.15))
    model.add(Dense(activation="relu", units=120))
    model.add(Dropout(0.15))
    model.add(Dense(activation="relu", units=120))
    model.add(Dropout(0.15))
    model.add(Dense(activation="softmax", units=3))
    opt = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse', optimizer=opt)
    model.load_weights("model.h5")
    return model

class Agent:
    
    def __init__(self):
        self.OBSERVE = OBSERVATION
        self.epsilon = INITIAL_EPSILON
        self.model = buildmodel()
        self.D = deque()
        self.t=0
    
    def setReward(self, app):
        reward = -0.01
        if not app._running:
            reward = -50
            return reward
        if app.snake.eaten:
            reward = 1000
        return reward


agent = Agent()
####FOOD####
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


####SNAKE####
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



#####GAME#####
class App:

    windowWidth = 330
    windowHeight = 330


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
    
    def getstate(self):
        state = [
                ((self.snake.dir == 1 and (pygame.Vector2(self.snake.pos.x+1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 25) or
                 (self.snake.dir == 2 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y-1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 0) or
                 (self.snake.dir == 3 and (pygame.Vector2(self.snake.pos.x-1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 0) or
                 (self.snake.dir == 4 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y+1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 25)
                 ), #obstacle front
                
                ((self.snake.dir == 1 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y-1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 0) or
                 (self.snake.dir == 2 and (pygame.Vector2(self.snake.pos.x-1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 0) or
                 (self.snake.dir == 3 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y+1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 25) or
                 (self.snake.dir == 4 and (pygame.Vector2(self.snake.pos.x+1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 25)
                 ), #obstacle left
                
                ((self.snake.dir == 1 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y+1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 25) or
                 (self.snake.dir == 2 and (pygame.Vector2(self.snake.pos.x-1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 0) or
                 (self.snake.dir == 3 and (pygame.Vector2(self.snake.pos.x,self.snake.pos.y-1) in self.snake.positions[:self.snake.length]) or self.snake.pos.y == 0) or
                 (self.snake.dir == 4 and (pygame.Vector2(self.snake.pos.x+1,self.snake.pos.y) in self.snake.positions[:self.snake.length]) or self.snake.pos.x == 25)
                 ), #obstacle right
                self.snake.dir == 1, #moving right
                self.snake.dir == 2, #moving up
                self.snake.dir == 3, #moving left
                self.snake.dir == 4, #moving down
                self.food.pos.x > self.snake.pos.x, #food is left of snake
                self.food.pos.x < self.snake.pos.x, #food is right of snake
                self.food.pos.y > self.snake.pos.y, #food is below snake
                self.food.pos.y < self.snake.pos.y  #food is above snake
                ]
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        state = np.asarray(state)
        state = np.expand_dims(state, axis=0)
        return np.asarray(state)
    
    
    
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
                    self._running = False
                    pygame.quit()
                    quit()
            
            agent.state = self.getstate()
            agent.loss = 0
            agent.Q_sa = 0
            agent.action_index = 0
            agent.reward = 0
            move = np.zeros([ACTIONS])
            #choose an action epsilon greedy
            if random.random() <= agent.epsilon:
                action_index = random.randrange(ACTIONS)
                move[action_index] = 1
            else:
                agent.q = agent.model.predict(agent.state)
                agent.max_Q = np.argmax(agent.q)
                action_index = agent.max_Q
                move[agent.max_Q] = 1
            
            if agent.epsilon > FINAL_EPSILON and agent.t > agent.OBSERVE:
                agent.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
#            if random.randint(0,100) < agent.epsilon:
#                move = to_categorical(random.randint(0,2), num_classes = 3)
#            else:
#                prediction = agent.model.predict(state_old.reshape((1,11)))
#                move = to_categorical(np.argmax(prediction[0]), num_classes = 3)
            
            self.snake.doMove(move)
            self.on_loop()
            
            agent.state_new = self.getstate()
            reward = agent.setReward(self)
            agent.D.append((agent.state, action_index, reward, agent.state_new, self._running))
            if len(agent.D) > REPLAY_MEMORY:
                agent.D.popleft()
            
#            agent.shortMemory(state_old, move, reward, state_new, self._running)
#            agent.remember(state_old, move, reward, state_new, self._running)
            record = getRecord(self.snake.length-4, record)
            
                #only train if done observing
            if agent.t > agent.OBSERVE:
                #sample a minibatch to train on
                minibatch = random.sample(agent.D, BATCH)
    
                #Now we do the experience replay
                state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
                state_t = np.concatenate(state_t)
                state_t1 = np.concatenate(state_t1)
                targets = agent.model.predict(state_t)
                agent.Q_sa = agent.model.predict(state_t1)
                targets[range(BATCH), action_t] = reward_t + GAMMA*np.max(agent.Q_sa, axis=1)*np.invert(self._running)
    
                agent.loss += agent.model.train_on_batch(state_t, targets)
    
            agent.state = agent.state_new
            agent.t = agent.t + 1
    
            # save progress every 10000 iterations
            if agent.t % 10000 == 0 and agent.t > agent.OBSERVE:
                print("Now we save model")
                agent.model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(agent.model.to_json(), outfile)
            
            self.on_render()

            time.sleep (50.0 / 1000.0);
        if agent.t < agent.OBSERVE:
            print("Observing: ",agent.t)
        self.on_cleanup()


if __name__ == "__main__" :
    while noOfGames < 1000:
        game = App()   
        game.on_execute()
        noOfGames += 1
    print("Now we save model")
    agent.model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(agent.model.to_json(), outfile)
    #agent.model.save_weights('weights.hdf5')
