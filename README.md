# EmotionalEmojis-PT
A series of scripts for creating datasets and model for emotion classification for Portuguese based on emojis/emotion relations. The contents of this directory were used for the creation of several computational models for my master thesis, available in Portuguese, which can be found [Here](https://estudogeral.sib.uc.pt/bitstream/10316/88059/1/Tese_LuisDuarteFinal.pdf).

# Repo contents:
1. datasets: Direcotry containing the created datasets used, due to legal issues we can't make them publicly available
2. Lexicons: Directory containing all the lexicons used, also due to legal issues only sharable links are displayed
   - [ANEW-PT](http://p-pal.di.uminho.pt/about/databases)
   - [Sentilex-PT](https://github.com/daviddias/METI-EADW/blob/master/src/sentimentAnalisys/SentiLex-lem-PT02.txt)
   - [NRC-PT](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
   - [LEED](https://osf.io/nua4x/)
3. word_emo2vec: Direcotry where saved word2vec models are stored
4. crawler.py: Script for text and emotion assignment of Twitter texts of last 7 days due to tweepy API limit
5. data_processing.py: Script for data cleanse process
6. elmo.py: Script of additional experiments using ELMO word embeddings. 
7. emotions.py: Script for dataset transformation to produce a new dataset where the texts are the ones with only one emoji and are placed at the end of the text/tweet
8. execute_all.py: Script for executing crawler.py file for all 6 Ekman basic emotions and all emojis.
9. model_definitions.py: Script for model creation and data processing for emotion classification from text. Main part of all the work is done in this script
10. streaming_crawler.py: Script for extracting tweets in real time and storage with automatic labelling.


# Citations:
All the contents in this repo were used like i previously mentioned on my master thesis which can be cited:

```
@phdthesis{duarte2019reconhecimento,
  title={Reconhecimento Autom{\'a}tico de Emo{\c{c}}{\~o}es em Texto com recurso a emojis},
  author={Duarte, Lu{\'\i}s Carlos Fernandes},
  year={2019},
  school={Universidade de Coimbra}
}
```
And another article of the same subject, published at EPIA 2019: 

```
@inproceedings{duarte2019exploring,
  title={Exploring Emojis for Emotion Recognition in Portuguese Text},
  author={Duarte, Luis and Macedo, Lu{\'\i}s and Oliveira, Hugo Gon{\c{c}}alo},
  booktitle={EPIA Conference on Artificial Intelligence},
  pages={719--730},
  year={2019},
  organization={Springer}
}
```
