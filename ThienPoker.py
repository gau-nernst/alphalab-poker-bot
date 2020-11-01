import tensorflow as tf
import numpy as np
from pypokerengine.players import BasePokerPlayer

INITIAL_STACK = 1000

suites = ('H', 'C', 'S', 'D')
faces = [x for x in range(2,10)] + ['T', 'J', 'Q', 'K', 'A']

encoder = {}
i = 0

for s in suites:
    for f in faces:
        encoder[f'{s}{f}'] = i
        i += 1

class ThienPlayer(BasePokerPlayer):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.stacks_history = np.ones((32,2))*INITIAL_STACK
        self.state_history = []
        self.action_history = []
        self.payoff_history = []
    
    def declare_action(self, valid_actions, hole_card, round_state):
        round_count = round_state['round_count'] % 32
        street = round_state['street']
        if street == 'preflop':
            return valid_actions[1]['action'], valid_actions[1]['amount']
        
        hole = [encoder[x] for x in hole_card]
        community = [encoder[x] for x in round_state['community_card']]
        community += [-1] * (5-len(community))
        
        hole = tf.one_hot(hole, 52)
        hole = tf.expand_dims(hole, axis=0)

        community = tf.one_hot(community, 52)
        community = tf.expand_dims(community, axis=0)
            
        for x in round_state['seats']:
            if x['name'] == self.name:
                self.stacks_history[round_count, 0] = x['stack']
                self.payoff_history.append(x['stack'] - self.stacks_history[(round_count-1+32)%32, 0]) 
            else:
                self.stacks_history[round_count, 1] = x['stack']
        
        stacks_history = np.vstack([self.stacks_history[round_count:], self.stacks_history[:round_count]])
        stacks_history = np.expand_dims(stacks_history, axis=0)
        stacks_history = tf.convert_to_tensor(stacks_history)
        
        state = [hole, community, stacks_history]
        self.state_history.append(state)
        
        action = self.model.predict(state)[0]
        
        action = np.random.choice(3, p=action)
        self.action_history.append(action)
        
        if action < 2:
            amount = valid_actions[action]['amount']
        else:
            amount = valid_actions[action]['amount']['max']//5
        
        return valid_actions[action]['action'], amount

    def receive_game_start_message(self, game_info):
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        pass
    
    def receive_street_start_message(self, street, round_state):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        pass
    
def create_model():
    hole_cards = tf.keras.Input(shape=(2,52,), name='hole_cards')
    community_cards = tf.keras.Input(shape=(5,52,), name='community_cards')
    stacks_history = tf.keras.Input(shape=(32,2,), name='stacks_history')
    
    cards = tf.keras.layers.concatenate([hole_cards, community_cards], axis=1)
    cards = tf.keras.layers.Flatten()(cards)
    stacks = tf.keras.layers.LSTM(32)(stacks_history)
    
    cards_dense = tf.keras.layers.Dense(50, activation='relu')(cards)
    cards_dense = tf.keras.layers.Dense(10, activation='relu')(cards)
    x = tf.keras.layers.concatenate([cards_dense, stacks])
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    x = tf.keras.layers.Dense(100, activation='relu')(x)
    
    action = tf.keras.layers.Dense(3, activation='softmax', name='action')(x)
    
    model = tf.keras.Model(
        inputs=[hole_cards, community_cards, stacks_history],
        outputs=action
    )
    
    return model