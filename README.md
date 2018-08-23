# Neural-CRF-AE
## Requirments

* PyTorch 0.3.0
* spaCy 2.0.0
* Python 3.6

## External Resources

Features are available at [Google Drive](https://drive.google.com/drive/folders/1aI2_nT3Aym1W8wimj8D1kLgX-t9H1sNu?usp=sharing)

Gazetteers are available at [Google Drive](https://drive.google.com/drive/folders/1XPLU70aLXh8VLfsDG8caPFNtomxwwYTq?usp=sharing)

## Instructions

1. Clone this repo.
2. Create three new folders ``models``, ``features`` and ``checkpoints``.
3. Download pre-trained word embeddings to ``models`` and feature files to ``features``.
4. Run ``python main.py`` and the model will be save at ``checkpoints``

## Acknowledgement

Some programs are adapted from:

* [Official PyTorch Tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)
* [A Theano implementation for Lample's work](https://github.com/glample/tagger)
* [A PyTorch implementation for Ma's work](https://github.com/ZhixiuYe/NER-pytorch)
* [A Keras implementation for Ma's work](https://github.com/UKPLab/emnlp2017-bilstm-cnn-crf)

Thank you for your contributions.
