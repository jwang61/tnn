import json
import re
import pickle

FILE = "tweets.json"

START_CHAR = "<"
END_CHAR = ">"

URL_REGEX = "[ ]*http[^ ]+[ ]*"
UNICODE_REGEX = "[^\x00-\x7F]"
REPLACE_UNICODE = "^"

WACKO_MAP = {8216 : "'",
             8217 : "'",
             180  : "'",
             8242  : "'",
             8211 : "-",
             8212 : "-",
             8213 : "-",
             8220 : "\"",
             8221 : "\"",
             243  : "o",
             201  : "E",
             233  : "e",
             232  : "e",
             333  : "o",
             7877 : "e",
             226  : "a",
             250  : "u",
             287  : "g",
             248  : "o",
             225  : "a",
             241  : "n",
             237  : "i",
             239  : "i",
             305  : "i",
             160  : " ",
             8230 : "...",
             8226 : "-",
             8252 : "!",
             8203 : " ",
             163  : "$",
             8364 : "$",
             8482 : "TM",
             174  : "",
             146  : "'",
             96   : "'",
             186  : "0",
             8206 : " ",
             8207 : " ",
             65330 : "R",
             65332 : "T",
             13   : "\n",
             12298: "~"}

EMOJIS = [10140, 9752, 9989, 9918, 9745, 9728, 9737, 10060, 10004, 11015, 10084, 314, 9786, 9734,
          9825, 9829, 9729, 9679, 11013, 171, 187, 9758, 9996, 10024, 9785, 9992, 9757, 9994, 9971,
          9733, 10145, 65039]

with open(FILE, 'r') as f:
    tweets = json.load(f)

char_vocab = set()
counter = 0
with open("tweets.txt", 'w') as f:
    for tweet in tweets:
        if tweet["is_retweet"] or tweet["text"].startswith("\"@"):
            continue

        tweet_fixed = tweet["text"].replace("&amp;", "&")

        tweet_fixed = re.sub(URL_REGEX, "", tweet_fixed)
        tweet_fixed = tweet_fixed.replace("\n\n", "\n")

        tweet_fixed = list(tweet_fixed)
        for i in range(len(tweet_fixed)):

            ordinal = ord(tweet_fixed[i])
            if ordinal in WACKO_MAP:
                tweet_fixed[i] = WACKO_MAP[ordinal]
                continue
            if ordinal > 127000 or ordinal in EMOJIS:
                tweet_fixed[i] = REPLACE_UNICODE

        tweet_fixed = "".join(tweet_fixed)
    
        if len(tweet_fixed) < 3:
            continue

        tweet_fixed = START_CHAR + tweet_fixed + END_CHAR

        char_vocab.update(tweet_fixed)
        f.write(tweet_fixed)
        counter += 1

with open("vocab.pkl", 'wb') as f:
    pickle.dump(char_vocab, f)
print(sorted(char_vocab))
print("Vocab Size: {}".format(len(char_vocab)))
print("Num Counts: {}".format(counter))
print("Done processing!")
