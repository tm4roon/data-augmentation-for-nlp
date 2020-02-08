# :construction:　Data Augmentation for Natural Language Processing

This code is an implementation of *data augmentation* for natural language processing tasks.  The data augmentation is an expanding training data method, which generates pseudo-sentences from supervised sentences.


## Installation

This code is depend on the following.

- python==3.6.5
- pytorch==1.1.0


```sh
git clone /path/to/this/repository
cd repository
pip install -r requirements.txt
```


## Data Augmentation

You can choose a data augmentation strategy, using a combination of a sampling strategy `--sampling-strategy` and a generation strategy `--augmentation-strategy`. 
<br>
 



### Sampling storategies (`--sampling-storategy`)

This option decides how to sample token's positions in original sentence pairs.

| storategy                  | description                                                                                  |
| ---------------------------| -------------------------------------------------------------------------------------------- |
| random                     | randomly sample tokens.                                                                      |
| absolute_discounting       |                                                                                              |
<br>


### Generation storategies


| storategy | description                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------- |
| dropout   | drop a token (Iyyer et al., 2015; Lample et al., 2017);                                                       |
| blank     | replace a token with a placeholder token (Xie et al., 2017);                                                  |
| smooth    | replace a token with a sample from the unigram frequency distribution over the vocabulary (Xie et al., 2017); |
| wordnet   |                                                                                                               |
| ppdb      |                                                                                                               |
| word2vec  |                                                                                                               |
| bert      |                                                                                                               |
<br>


## Replacing probability scheduling





## Usage





## Options





## References

