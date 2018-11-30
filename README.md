# tnn
Twitter Neural Network: char-rnn for tweets.
Train and sample an LSTM model using tensorflow.

Dataset used: Trump's tweets found at: http://www.trumptwitterarchive.com/

# Requirements
```
python == 3.6.5
numpy >= 1.15.2
tensorflow >= 1.11.0
```

# Scripts

After downloading all the tweets, the script `python3 process_tweets.py` is used to remove unnecessary data, standardize characters by removing emojis and accents, and remove retweets and really short tweets.

Call `python3 train.py` to train the neural network and `python3 sample.py` to produce samples once the network is trained.

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
