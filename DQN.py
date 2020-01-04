import keras
import random
import pygame
import numpy as np

class Agent():
    
    def __init__(self):
        self.reward = 0
        self.gamma = 0.9
        self.target = 1
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = 0.005
        self.model = self.network()
        self.epsilon = 80
        self.actual = []
        self.memory = []
    
    def getState(self, app):
        state = [
                ((app.snake.dir == 1 and (pygame.Vector2(app.snake.pos.x+1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 25) or
                 (app.snake.dir == 2 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y-1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 0) or
                 (app.snake.dir == 3 and (pygame.Vector2(app.snake.pos.x-1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 0) or
                 (app.snake.dir == 4 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y+1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 25)
                 ), #obstacle front
                
                ((app.snake.dir == 1 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y-1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 0) or
                 (app.snake.dir == 2 and (pygame.Vector2(app.snake.pos.x-1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 0) or
                 (app.snake.dir == 3 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y+1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 25) or
                 (app.snake.dir == 4 and (pygame.Vector2(app.snake.pos.x+1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 25)
                 ), #obstacle left
                
                ((app.snake.dir == 1 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y+1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 25) or
                 (app.snake.dir == 2 and (pygame.Vector2(app.snake.pos.x-1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 0) or
                 (app.snake.dir == 3 and (pygame.Vector2(app.snake.pos.x,app.snake.pos.y-1) in app.snake.positions[:app.snake.length]) or app.snake.pos.y == 0) or
                 (app.snake.dir == 4 and (pygame.Vector2(app.snake.pos.x+1,app.snake.pos.y) in app.snake.positions[:app.snake.length]) or app.snake.pos.x == 25)
                 ), #obstacle right
                app.snake.dir == 1, #moving right
                app.snake.dir == 2, #moving up
                app.snake.dir == 3, #moving left
                app.snake.dir == 4, #moving down
                app.food.pos.x > app.snake.pos.x, #food is left of snake
                app.food.pos.x < app.snake.pos.x, #food is right of snake
                app.food.pos.y > app.snake.pos.y, #food is below snake
                app.food.pos.y < app.snake.pos.y  #food is above snake
                ]
        for i in range(len(state)):
            if state[i]:
                state[i]=1
            else:
                state[i]=0
        
        return np.asarray(state)

    def network(self, weights=None):
        model = keras.models.Sequential()
        model.add(keras.layers.core.Dense(activation="relu", input_dim=11, units=120))
        model.add(keras.layers.core.Dropout(0.15))
        model.add(keras.layers.core.Dense(activation="relu", units=120))
        model.add(keras.layers.core.Dropout(0.15))
        model.add(keras.layers.core.Dense(activation="relu", units=120))
        model.add(keras.layers.core.Dropout(0.15))
        model.add(keras.layers.core.Dense(activation="softmax", units=3))
        opt = keras.optimizers.Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model
    
    def setReward(self, app):
        self.reward = -0.01
        if not app._running:
            self.reward = -500
            return self.reward
        if app.snake.eaten:
            self.reward = 100
        return self.reward
    
    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

    def replay(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

    def shortMemory(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, 11)))[0])
        target_f = self.model.predict(state.reshape((1, 11)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, 11)), target_f, epochs=1, verbose=0)