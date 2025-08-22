# Introduction

This repository contains the code necessary to reproduce UW-BENMT's system submission to WMT2025 Terminology Shared Task. The products are Neural Machine Translation models that are able to translate information-technology content, with special focus on translating terminologies in this domain, across en-de, en-ru and en-es language pairs. In particular, this system addresses a research gap in terminology-aware neural machine translation - how does one select terminology constraints as soft-constraints at training time?

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
   
   git clone https://github.com/Benjamin-Pong/awesome-align.git
   
   conda env create -f environment.yml

   conda activate awesome_align
   
3. Fairseq for NMT training:
   
   git clone https://github.com/Benjamin-Pong/fairseq.git
   
   conda env create -f environment.yml
   
   conda activate fairseq



