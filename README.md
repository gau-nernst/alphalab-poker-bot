# alphalab-poker-bot

## Dependencies
- PyPokerEngine
- TensorFlow
- Numpy

(For visualization only)
- Matplotlib
- Seaborn

Create virtual environment with `conda`

```
conda create -n ThienPoker python=3.8
conda activate ThienPoker
pip install PyPokerEngine
conda install tensorflow-gpu numpy=1.19
```

## Submission

To use the bot for submission, import the function `get_submssion` from `submission.py`. This function will return a Player object for use with PyPokerEngine.

Note that due to limitation in my implementation, the name of the player must be `p1` when register with the PyPokerEngine game config.

```python
from submission import get_submission
poker_player = get_submission()

config = setup_config(max_round=10, initial_stack=INITIAL_STACK, small_blind_amount=5)
config.register_player(name="p1", algorithm=poker_player)
config.register_player(name="p2", algorithm=FishPlayer())
game_result = start_poker(config, verbose=1)
game_result
```

## Model training

The training process can be found in the notebook `poker.ipynb`. The bot was trained entirely by playing with itself for 1000 epochs.

## Use the poker bot

An example on how to load the trained poker bot and play with FishPlayer, HonestPlayer and RandomPlayer can be found in the notebook `use_poker_bot.ipynb` 

## Brief explanation

Model definition can be found in `ThienPoker.py`. This file contains 2 things:
- Class `ThienPlayer`: Create a subclass of PyPokerEngine's `BasePokerPlayer` to play poker. Require the TensorFlow model defined below
- Function `create_model()`: Create a TensorFlow model to process which action to do

The TensorFlow model takes 3 inputs:
- Hole cards from `round_state`
- Community cards from `round_state`
- Historical stacks for each player

And it will output 3 numbers corresponding to the probabilities to take each action FOLD, CALL or RAISE
- `ThienPlayer` will take 1 out of the 3 actions based on the probabilities (using `np.random.choice()`)
- The RAISE amount is fixed to max raise amount divided by 5 in `ThienPlayer`. Initially I wanted to have another output head to predict the RAISE AMOUNT, but I haven't figured out how to use custom loss function for 2-output TensorFlow model, so I don't do that for now.

## Model architecture

Input representation
- One-hot encoding for each card. Thus hole cards will have shape `(2,52)` and community cards will have shape `(5,52)` (zero-padding until 5 cards)
- Historical stacks keep track of the last 32 stack amount of its own and the opponent. The shape is `(32,2)`

Processing
- Hole cards and community cards inputs are concatenated together to shape `(7,52)` and flattened to `(364)`
- Since this is a sparse representation, add a Dense layer of 10 neurons after this to condense the information from the cards. The idea is that this layer represents the strength of the hand.
- Historical stacks are passed through an LSTM layer. The idea is to extract out any information from the historical stack amounts.
- The two layers above are concatenated together
- A Dense output layer with 3 neurons is connected to provide the probabilities output

Some issues I am aware of:
- Currently the bot is built to play against 1 opponent only (2-player poker game). The initial stack is set to 1000
- The bot does not take into account the past actions of the opponent nor itself. I wanted to implement some kind of regret matching as input to the ML model, but couldn't do so due to time constraint.
- From my testing, on average, the bot losses to FishPlayer, HonestPlayer and RandomPlayer. So the performance of the bot is still not very good. 