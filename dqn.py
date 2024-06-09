from connect4 import create_board, print_board, draw_board, is_valid_location, get_next_open_row, drop_piece, winning_move, get_available_moves, BLACK, RED, YELLOW, GREEN, SQUARESIZE, RADIUS, width, size, ROW_COUNT, COLUMN_COUNT
import jax
from jax import numpy as jnp
from jax.lax import conv_general_dilated
from random import choice
from jax import random
import numpy as np
from functools import partial

class MyAdam:

    def __init__(self, lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, eps = 1e-8,):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
    
    def init_state(self, params):
        return dict(m=jax.tree_map(lambda p: 0, params), v=jax.tree_map(lambda p: 0, params), t=0)
    
    def update(self, grads, state):
        t = state["t"] + 1
        mt = jax.tree_map(lambda m, g: (self.beta_1*m) + ((1-self.beta_1)*g), state["m"], grads)
        vt = jax.tree_map(lambda v, g: (self.beta_2*v) + ((1-self.beta_2)*(g**2)), state["v"], grads)
        dparams = jax.tree_map(lambda m, v: self.lr*(m/(1-(self.beta_1**t)))/(self.eps + (jnp.sqrt(v/(1-(self.beta_2**t))))), mt, vt)

        return dparams, dict(m=mt, v=vt, t=t)

class MySGD:
    def __init__(self, lr = 0.1):
        self.lr = lr
    
    def init_state(self, params):
        return dict()

    def update(self, grads, state):
        dparams = jax.tree_map(lambda g: self.lr*g, grads)
        return dparams, dict()

class ReplayBuffer:

    def __init__(self, capacity, observation_shape, batch_size = 32):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.batch_size = batch_size

    def init_buffer(self):
        buffer = dict()
        buffer['states'] = jnp.zeros((self.capacity,) + self.observation_shape, dtype=jnp.float32)
        buffer['actions'] = jnp.zeros(self.capacity, dtype=jnp.int32)
        buffer['rewards'] = jnp.zeros(self.capacity, dtype=jnp.float32)
        buffer['next_states'] = jnp.zeros((self.capacity,) + self.observation_shape, dtype=jnp.float32)
        buffer['dones'] = jnp.zeros(self.capacity, dtype=jnp.bool)
        buffer['index'] = 0
        buffer['size'] = 0

        return buffer
    
    @partial(jax.jit, static_argnums=0)
    def add(self, buffer, state, action, reward, next_state, done):
        new_buffer = dict()
        new_buffer['states'] = buffer['states'].at[buffer['index']].set(state)
        new_buffer['actions'] = buffer['actions'].at[buffer['index']].set(action)
        new_buffer['rewards'] = buffer['rewards'].at[buffer['index']].set(reward)
        new_buffer['next_states'] = buffer['next_states'].at[buffer['index']].set(next_state)
        new_buffer['dones'] = buffer['dones'].at[buffer['index']].set(done)
        
        new_buffer['index'] = (buffer['index'] + 1) % self.capacity
        new_buffer['size'] = jnp.minimum(buffer['size'] + 1, self.capacity)

        return new_buffer

    @partial(jax.jit, static_argnums=0)
    def sample(self, key, buffer):
        indices = random.choice(key, jnp.arange(self.capacity), shape=(self.batch_size,), replace=False)
        return buffer['states'][indices], buffer['actions'][indices], buffer['rewards'][indices], buffer['next_states'][indices], buffer['dones'][indices]

class DQN:

    def __init__(self, key, input_dims=4, output_dims=2, layers=[]):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layers = layers
        self.params, key = self.init_params(key)

    def init_params(self, key):
        params = []

        layers = [self.input_dims] + self.layers + [self.output_dims]

        for layerIn, layerOut in zip(layers, layers[1:]):
            param = dict()
            key, keyW, keyb = random.split(key, num=3)
            param['W'] = 0.1*random.normal(keyW, (layerIn, layerOut))
            param['b'] = 0.1*random.normal(keyb, (layerOut,))
            params.append(param)

        return params, key

    @staticmethod
    def apply(params, x):
        for layer in params[:-1]:
            x = x@layer['W'] + layer['b']
            x = jnp.maximum(x, 0.)
        x = x@params[-1]['W'] + params[-1]['b']
        return x 

class DQNAgent:

    def __init__(self, seed = 42, epsilon = 0.1, input_dims=4, output_dims=2, network_layers = [100], lr = 0.01, capacity = 100):
        self.seed = 42
        self.epsilon = epsilon
        self.key = random.PRNGKey(seed)
        self.key, keyQ = random.split(self.key)
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.q_network = DQN(keyQ, input_dims=input_dims, output_dims=output_dims, layers=network_layers)
        self.q_target = DQN(keyQ, input_dims=input_dims, output_dims=output_dims, layers=network_layers)
        
        self.capacity = capacity

