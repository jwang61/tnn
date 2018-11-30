# tnn
Twitter Neural Network: character level LSTM model for generating tweets.
Dataset trained on: [Trump Twitter Archive](http://www.trumptwitterarchive.com)

This was inspired by [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow) and [char-rnn](https://github.com/karpathy/char-rnn).

# Requirements
- python == 3.6.5
- numpy >= 1.15.2
- tensorflow >= 1.11.0

# Scripts

After downloading all the Trump tweets, the script `python3 process_tweets.py` is used to:
- Remove all the unnecessary json values
- Remove emojis and standardize character vocabulary
- Remove short tweets (length < 5)
- Add "<" and ">" to mark start and end of tweet

Use `python3 train.py` to train the neural network. Model parameters can be tweaked in `model.py`
Use `python3 sample.py` to produce samples once the network is trained.

# Sample Output
With the feed "MAGA":
```
MAGA America must fight to protect the American People. Will be interviewed by the United States Supreme Court. I will be at the same time the only great job, the White House was a wonderful shape. I will be smart and not asked for our disgraceful criminal aliens - and every day back!
```

With the feed "Crook":
```
Crooked Hillary Clinton is considered a fraud with the majority. Think about her new season.
```

With the feed "Barack":
```
Barack Obama release his president who is staying through the world recent ratings.  #TimeToGetTough
```

With no feed:
```
.@TrumpNewYork's #CNNDebate @TrumpChicago's wedd is 'Friends of Chicago's Spa
```

# Future
TODO: Add argument parser to scripts, different models
