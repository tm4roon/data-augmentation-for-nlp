# Data Augmentation for Natural Language Processing

This code is an implementation of *data augmentation* for natural language processing tasks.  The data augmentation is an expanding training data method, which generates pseudo-sentences from supervised sentences.


## Installation

This code is depend on the following.

- python>=3.6.5


```sh
git clone https://github.com/tkmaroon/data-augmentation-for-nlp.git
cd repository
pip install -r requirements.txt
```



## Data Augmentation

You can choose a data augmentation strategy, using a combination of a sampling strategy `--sampling-strategy` and a generation strategy `--augmentation-strategy`. 
<br>




### Sampling storategies (`--sampling-storategy`)

This option decides how to sample token's positions in original sentence pairs.

| storategy | description             |
| --------- | ----------------------- |
| random    | randomly sample tokens. |
| -         | -                       |

<br>


### Generation storategies


| storategy | description                                                  |
| --------- | ------------------------------------------------------------ |
| dropout   | Drop a token [1, 2];                                         |
| blank     | Replace a token with a placeholder token [3];                |
| unigram   | Replace a token with a sample from the unigram frequency distribution over the vocabulary [3]. Please set the option `--unigram-frequency-for-generation`. [3]; |
| bigramkn  | Bigram Kneser-Ney smoothing [3]. Please set the option `--bigram-frequency-for-generation`. |
| wordnet   | Replace a token with a synonym of wordnet. Please set the option `--lang-for-wordnet`. |
| ppdb      | Replace a token with a paraphrase by given paraprase database. Please set the option `--ppdb-file`. |
| word2vec  | Replace a token with a token which has similar vector of word2vec. Please set the option `--w2v-file`. |
| bert      | Replace a token using output probability of BERT mask token prediction. Please set the option `--model-name-or-path`. It must be in the shortcut name list of hugging face's [pytorch-transformers](https://huggingface.co/transformers/). Note that the option `--vocab-file` must be same the vocabulary file of a BERT tokenizer. |
| -         | -                                                            |

<br>



## Usage

```sh
python generate.py \
    --input ./data/sample/sample.txt \
    --augmentation-strategy bert \
    --model-name-or-path bert-base-multilingual-uncased \
    --temparature 1.0
```





## References

[1] [Mohit Iyyer, Varun Manjunatha, Jordan Boyd-Graber, and Hal Daume III. Deep unordered com- ´ position rivals syntactic methods for text classification. ACL2015, volume 1, pages 1681–1691.](https://www.aclweb.org/anthology/P15-1162/)

[2] [Guillaume Lample, Alexis Conneau, Ludovic Denoyer, and Marc’Aurelio Ranzato. 2017. Unsupervised machine translation using monolingual corpora only. arXiv preprint arXiv:1711.00043.](https://arxiv.org/abs/1711.00043)

[3] [Ziang Xie, Sida I Wang, Jiwei Li, Daniel Levy, Aiming ´ Nie, Dan Jurafsky, and Andrew Y Ng. 2017. Data noising as smoothing in neural network language models. arXiv preprint arXiv:1703.02573.](https://arxiv.org/abs/1703.02573)

