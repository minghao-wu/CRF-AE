
import torch
from collections import OrderedDict

parameters = OrderedDict()
parameters['tag_scheme'] = "iobes"
parameters['lower'] = True
parameters['zeros'] = False
parameters['char_dim'] = 25
parameters['char_lstm_dim'] = 25
parameters['char_bidirect'] = True
parameters['word_dim'] = 300
parameters['word_lstm_dim'] = 300
parameters['word_bidirect'] = True
parameters['all_emb'] = True
parameters['cap_dim'] = 0
parameters['crf'] = True
parameters['dropout'] = 0.5
parameters['reload'] = False
parameters['name'] = "test"
parameters['char_mode'] = "CNN"
parameters["train"] = "dataset/eng.train"
parameters["dev"] = "dataset/eng.testa"
parameters["test"] = "dataset/eng.testb"
parameters["test_train"] = "dataset/eng.train54019"
parameters["pre_emb"] = "models/glove.6B.300d.txt"
parameters['use_gpu'] = torch.cuda.is_available()
parameters["features_dim"] = 196
parameters["feature_train"] = "features/all_onehot.train"
parameters["feature_dev"] = "features/all_onehot.testa"
parameters["feature_test"] = "features/all_onehot.testb"

parameters["gazetter_dim"] = 3
parameters["gazetteer_train"] = "features/gazetteer_PERLOC.train"
parameters["gazetteer_dev"] = "features/gazetteer_PERLOC.testa"
parameters["gazetteer_test"] = "features/gazetteer_PERLOC.testb"

parameters["pos_lambda"] = 1
parameters["wordshape_lambda"] = 1
parameters["gazetteer_lambda"] = 1


parameters["learning_rate"] = 0.015
parameters["epochs"] = 35
