# wordle_machine
Learn just enough about Wordle to get better, but not enough to ruin the fun

# Word Space
Starting with a base of the 20k most common words on the internet, this trims down those words by taking
1. Take all five letter words
2. Remove plurals as they are not in the history (with a rare exception)
3. Remove proper nouns
4. Keep only words that are identical to their relevant dictionary entry. This omits conjugated verbs ("heard" would be "hear" in the dictionary, so it is not valid).

Generates analysis of remaining words to inform guesses:
<img width="1157" alt="IMG_0342" src="https://user-images.githubusercontent.com/12261655/155397471-defdf467-b7a8-48f4-a77a-08e3095289f8.PNG">

Also generates starting words that will, on average, create the greatest reduction in the space of possible words. The word with the median lowest guessing space is...
**NOISE**
