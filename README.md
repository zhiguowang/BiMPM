# BiMPM: Bilateral Multi-Perspective Matching for Natural Language Sentences

## Description
This repository includes the source code for natural language sentence matching. 
Basically, the program will take two sentences as input, and predict a label for the two input sentences. 
You can use this program to deal with tasks like [paraphrase identification](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_%28State_of_the_art%29), [natural language inference](http://nlp.stanford.edu/projects/snli/), [duplicate questions identification](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) et al. More details about the underneath model can be found in our [paper](https://arxiv.org/pdf/1702.03814.pdf). Please cite our paper when you use this program! :heart_eyes:

## Requirements
* python 2.7
* tensorflow 0.12

## Data format
Both the train and test set require a tab-separated format.
Each line in the train (or test) file corresponds to an instance, and it should be arranged as
> label	sentence#1	sentence#2	other_info

For more details about the data format, you can download the [Quora Question Pair](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing) dataset used in our [paper](https://arxiv.org/pdf/1702.03814.pdf).


## Training
You can find the training script at BiMPM/src/SentenceMatchTrainer.py

To see all the **optional arguments**, just run
> python BiMPM/src/SentenceMatchTrainer.py --help

Here is an example of how to train a very simple model:
> python  BiMPM/src/SentenceMatchTrainer.py --train\_path train.tsv --dev\_path dev.tsv --test\_path test.tsv --word\_vec_path wordvec.txt --suffix sample --fix\_word\_vec --model\_dir models/ --MP\_dim 20 

To get a better performance on your own datasets, you need to play with other arguments.

## Testing
You can find the testing script at BiMPM/src/SentenceMatchDecoder.py


To see all the **optional arguments**, just run
> python BiMPM/src/SentenceMatchDecoder.py --help

Here is an example of how to test your model:
> python  BiMPM/src/SentenceMatchDecoder.py --in\_path test.tsv --word\_vec\_path wordvec.txt --mode prediction --model\_prefix models/SentenceMatch.sample --out\_path test.prediction

The SentenceMatchDecoder.py can run in two modes:
* prediction: predicting the label for each sentence pair
* probs: outputting probabilities of all labels for each sentence pair

## Reporting issues
Please let [me](https://zhiguowang.github.io/) know, if you encounter any problems.
