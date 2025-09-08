# Introduction

This repository contains the code necessary to reproduce UW-BENMT's system submission to WMT2025 Terminology Shared Task. The products of the pipeline are Neural Machine Translation models that are able to translate information-technology content, with special focus on translating terminologies in this domain, across three language pairs; en-de, en-ru and en-es. In particular, this system addresses a research gap in terminology-aware neural machine translation - how does one select terminology constraints as soft-constraints at training time?

Practical applications of this system involve translating API documentations, code and technical manuals (i.e. DevOps manuals), allowing knowledge exchange across multilingual cross-functional operations.



# Overview of System

1. Data selection for IT-domain specific NMT training using Cross-Entropy Difference
2. Source-target word alignment using an off-the-shelf Neural Word Aligner - Awesome Align
3. Data processing to account for multiword alignments between source and target language
4. Select the most "important" word-alignment(s) for downstream terminology-aware NMT training by computing the norm of the encoder's final hidden states.
5. Evaluation of systems for general translation quality (BLEU, COMET, Chrff++), and Terminology Success Rate (with and without lemmatization)


# Getting Started

Install the respective dependencies to use the following infrastructures:

1. Awesome-align for Neural Word Alignment:

   ```
   git clone https://github.com/Benjamin-Pong/awesome-align.git
   ```
   
   ```
   conda env create -f environment.yml
   ```


   
3. Fairseq for NMT training:

   
   ```
   git clone https://github.com/Benjamin-Pong/fairseq.git
   ```

   ```
   conda env create -f environment.yml
   ```

# Preparing IT-related data for NMT training


To generate IT-related data, we will use the Moore-Lewis Cross-Entropy Difference approach to retrieve IT-related data from a pool of publicly available parallel data.

Sources of Publicly available data for each language pair:
- en-es, en-de: Europarl and WikiMatrix
- en-ru: WikiMatrix and Paracrawl

Download them into your local directory

## Generate in-domain data via synthetic data generation

Run the script synthetic_data_generation.ipynb using google colab. Since this uses an LLM to generate synthetic data, a gpu is required - L4.

Output would be three sets of synthetic data, one for spanish, one for german and one for russian. These files will serve as argument inputs into lm.py when we train our  in-domain statistical language models.

## Generate out-domain data

## Train in-domain and out-domain statistical language models

```bash
./lm.sh
```

Outputs are in-domain and out-domain language models (pickle files)

## Data selection by Cross-Entropy Difference scoring

```bash
./compute_CE.sh
```

Output is a corpora that has been ranked in non-increasing order based on the Cross-Entropy Difference

# Neural Word Alignment

The next step is to extract word-alignments from the data that we have selected for the parallel corpora. This is where Awesome Align will be integrated.

```bash
cd nwa
```

```
conda activate awesome_align
```

Convert the parallel data into the desired format to be consumed by the neural word aligner.

```bash
./prepare.sh
```

Next, activate your conda environment for awesome align, and run the following, where $DATA_FILE is the directory pointing to the output of ./prepare.sh.

```bash
cd  awesome-align directory 
CUDA_VISIBLE_DEVICES=0 awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --batch_size 64 \
    --output_word_file=$OUTPUT_WORD_FILE \
    --output_prob_file=$OUTPUT_PROB_FILE \
    --num_workers 2
```
# TBC




