# BiMPM: Bilateral Multi-Perspective Matching for Natural Language Sentences

## Updates (Jan 28, 2018)
* This repository has been updated to tensorflow 1.5
* The training process speeds up 15+ times without lossing the accuracy.
* All codes have been re-constructed for better readability and adaptability.

## Description
This repository includes the source code for natural language sentence matching. 
Basically, the program takes two sentences as input, and predict a label for the two input sentences. 
You can use this program to deal with tasks like [paraphrase identification](https://aclweb.org/aclwiki/index.php?title=Paraphrase_Identification_%28State_of_the_art%29), [natural language inference](http://nlp.stanford.edu/projects/snli/), [duplicate questions identification](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs) et al. More details about the underneath model can be found in our [paper](https://arxiv.org/pdf/1702.03814.pdf) published in IJCAI 2017. Please cite our paper when you use this program! :heart_eyes:

## Requirements
* python 2.7
* tensorflow 1.5

## Data format
Both the train and test sets require a tab-separated format.
Each line in the train (or test) file corresponds to an instance, and it should be arranged as
> label	sentence#1	sentence#2 instanceID	

For more details about the data format, you can download the [SNLI](https://drive.google.com/file/d/1CxjKsaM6YgZPRKmJhNn7WcIC3gISehcS/view?usp=sharing) and the [Quora Question Pair](https://drive.google.com/file/d/0B0PlTAo--BnaQWlsZl9FZ3l1c28/view?usp=sharing) datasets used in our [paper](https://arxiv.org/pdf/1702.03814.pdf).


## Training
You can find the training script at BiMPM/src/SentenceMatchTrainer.py

First, edit the configuration file at ${workspace}/BiMPM/configs/snli.sample.config (or ${workspace}/BiMPM/configs/quora.sample.config ).
You need to change the "train\_path", "dev\_path", "word\_vec\_path", "model\_dir", "suffix" to your own setting.

Second, launch job using the following command line
> python  ${workspace}/BiMPM/SentenceMatchTrainer.py --config\_path ${workspace}/BiMPM/configs/snli.sample.config


## Testing
You can find the testing script at BiMPM/src/SentenceMatchDecoder.py
> python  ${workspace}/BiMPM/src/SentenceMatchDecoder.py --in\_path ${your\_path\_to}/dev.tsv --word\_vec\_path ${your\_path\_to}/wordvec.txt --out\_path ${your\_path\_to}/result.json --model\_prefix ${model\_dir}/SentenceMatch.${suffix}

Where "model\_dir" and "suffix" are the variables set in your configuration file.

The output file is a json file with the follwing format.

```javascript
{
    { 
        "ID": "instanceID",
        "truth": label,
        "sent1": sentence1,
        "sent2": sentence2,
        "prediction": prediciton,
        "probs": probs_for_all_possible_labels
    },
    { 
        "ID": "instanceID",
        "truth": label,
        "sent1": sentence1,
        "sent2": sentence2,
        "prediction": prediciton,
        "probs": probs_for_all_possible_labels
    }
}
```


## Reporting issues
Please let [me](https://zhiguowang.github.io/) know, if you encounter any problems.
