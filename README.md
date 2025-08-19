# Overview

This repository contains the code necessary to reproduce UW-BENMT's system submission to WMT2025 Terminology Shared Task. The products are Neural Machine Translation models that are able to translate information-technology content, with special focus on translating terminologies in this domain, across en-de, en-ru and en-es language pairs.


# Getting Started

1. Awesome-align for Neural Word Alignment:
   
   git clone https://github.com/Benjamin-Pong/awesome-align.git
   
   conda env create -f environment.yml
   
   conda activate awesome_align
   
3. Fairseq for NMT training:
   
   git clone https://github.com/Benjamin-Pong/fairseq.git
   
   conda env create -f environment.yml
   
   conda activate fairseq
