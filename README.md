# Natural-Language-Generation

Code to natural language generation as defined in [this paper](./natural-language-generation-with-markov-chains-and -recurrent-neural-networks.pdf). 

## Abstract
This paper introduce two approaches to achieve natural language generation
using Markov Chains and Recurrent Neural Networks. Markov Chains uses a
probabilistic model based on the relationship between each unique word to calculate
the probability of the next word, which can be used to text generation. Recurrent
Neural Networks are powerful sequence models that are able to remember and
process the sequence of each input, which are popular used to solve text generation
problem. This paper will show the difference of two techniques through an experiment which
uses the works of William Shakespeare to generate texts with Shakespeare’s writing
style. This study shows that texts generated using Recurrent Neural Network are
better than texts using Markov Chains.

## Prerequisites

- python3.7
- numpy
- nltk

## Usage

```
python train.py
```
