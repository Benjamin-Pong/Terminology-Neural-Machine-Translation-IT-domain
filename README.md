# Introduction

This repository contains the code necessary to reproduce UW-BENMT's system submission to WMT2025 Terminology Shared Task. The products are Neural Machine Translation models that are able to translate information-technology content, with special focus on translating terminologies in this domain, across en-de, en-ru and en-es language pairs.

# Overview of System

1. Data selection for IT-domain specific NMT training using Cross-Entropy Difference
2. Source-target word alignment using an off-the-shelf Neural Word Aligner - Awesome Align
3. Data processing to account for multiword alignments between source and target language
4. Retrieval of most "important" word-alignments for downstream terminology-aware NMT training
5. Evaluation of systems for general translation quality (BLEU, COMET, Chrff++), and Terminology Success Rate (with and without lemmatization)


# Getting Started

1. Awesome-align for Neural Word Alignment:
   
   git clone https://github.com/Benjamin-Pong/awesome-align.git
   
   conda env create -f environment.yml
   
   conda activate awesome_align
   
3. Fairseq for NMT training:
   
   git clone https://github.com/Benjamin-Pong/fairseq.git
   
   conda env create -f environment.yml
   
   conda activate fairseq



