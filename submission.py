import tensorflow as tf
import numpy as np
from ThienPoker import ThienPlayer, create_model

submit_weights = 'saved_weights/thienplayer_20201101h134558'

def get_submission(name='p1'):
    poker_model = create_model()
    poker_model.load_weights(submit_weights)
    
    poker_player = ThienPlayer(name, poker_model)
    return poker_player