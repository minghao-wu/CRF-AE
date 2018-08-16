import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from torch import optim
from tqdm import tqdm

import optparse
import itertools
import loader

import time
import numpy as np
import datetime
import pickle
from torch.autograd import Variable
import matplotlib.pyplot as plt
import sys
from utils import *
from loader import *
from model import Neural_CRF_AE
from config import parameters

use_gpu = torch.cuda.is_available()
# if use_gpu:
#     GPU_id = 7
#     print("GPU ID = ", GPU_id)
#     torch.cuda.set_device(GPU_id)

print(use_gpu)




mapping_file = 'models/mapping.pkl'
print(parameters)
eval_script = "./evaluation/conlleval.pl"
eval_temp = "./evaluation/temp"
print("eval_script = ", eval_script)
print("eval_temp = ", eval_temp)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(parameters["train"], lower, zeros)
dev_sentences = loader.load_sentences(parameters["dev"], lower, zeros)
test_sentences = loader.load_sentences(parameters["test"], lower, zeros)
test_train_sentences = loader.load_sentences(parameters["test_train"], lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences, lower)[0]
dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )

dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

data = {
    "train":train_data,
    "dev":dev_data,
    "test":test_data
}
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(parameters["pre_emb"], 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), parameters["word_dim"]))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    pickle.dump(mappings, f)

print('word_to_id: ', len(word_to_id))

def evaluating(model, datas, best_F, features, gazetteer):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in tqdm(datas):
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']
        hands = data["handcrafted"]
        feature = features[min(hands):(max(hands)+1)]
        gaze = gazetteer[min(hands):(max(hands)+1)]
        feature = feature[:, 45:]

        if parameters['char_mode'] == 'CNN':
            d = {}
            chars2_length = [len(c) for c in chars2]
            char_maxl = max(chars2_length)
            chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
            for i, c in enumerate(chars2):
                chars2_mask[i, :chars2_length[i]] = c
            chars2_mask = Variable(torch.LongTensor(chars2_mask))

        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))
        feature = Variable(torch.FloatTensor(feature))
        gaze = Variable(torch.FloatTensor(gaze))
        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(), chars2_length, d, feature.cuda(), gaze.cuda())
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d, feature, gaze)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')

    eval_script = "./evaluation/conlleval.pl"
    eval_temp = "./evaluation/temp"

    predf = eval_temp + '/pred.test'
    scoref = eval_temp + '/score.test'

    with open(predf, 'w') as f:
        f.write('\n'.join(prediction))

    os.system('perl %s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    print(best_F, new_F, save)
    return best_F, new_F, save


# In[5]:


def train_model(model, dataset, optimizer, scheduler, num_epochs):
    sizes = {
        "train": len(dataset["train"]),
        "dev": len(dataset["dev"]),
        "test": len(dataset["test"])
    }
    splits = ["train", "dev", "test"]
    best_dev_F = -1.0
    best_test_F = -1.0
    train_count = 0
    dev_count = 0
    test_count = 0

    with open(parameters["feature_train"], "rb") as f:
        features_train = pickle.load(f)
    with open(parameters["feature_dev"], "rb") as f:
        features_dev = pickle.load(f)
    with open(parameters["feature_test"], "rb") as f:
        features_test = pickle.load(f)

    with open(parameters["gazetteer_train"], "rb") as f:
        gaze_train = pickle.load(f)
    with open(parameters["gazetteer_dev"], "rb") as f:
        gaze_dev = pickle.load(f)
    with open(parameters["gazetteer_test"], "rb") as f:
        gaze_test = pickle.load(f)

    for epoch in range(num_epochs):
        print("===========================================")
        print("Epoch %d / %d" % (epoch+1, num_epochs))
        loss = 0
        for phase in splits:
            if phase == "train":
                scheduler.step()
                model.train()
                random.shuffle(train_data)

                for data in tqdm(train_data):
                    train_count += 1
                    model.zero_grad()

                    sentence_in = data['words']
                    tags = data['tags']
                    chars2 = data['chars']
                    hands = data["handcrafted"]
                    feature = features_train[min(hands):(max(hands)+1)]
                    gaze = gaze_train[min(hands):(max(hands)+1)]

                    deps = feature[:,:45]
                    shapes = feature[:,45:196]
                    pos = feature[:,196:]
                    feature = feature[:, 45:]
                    # ######## char cnn
                    if parameters['char_mode'] == 'CNN':
                        d = {}
                        chars2_length = [len(c) for c in chars2]
                        char_maxl = max(chars2_length)
                        chars2_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
                        for i, c in enumerate(chars2):
                            chars2_mask[i, :chars2_length[i]] = c
                        if use_gpu:
                            chars2_mask = Variable(torch.LongTensor(chars2_mask).cuda())
                        else:
                            chars2_mask = Variable(torch.LongTensor(chars2_mask))

                    if use_gpu:
                        sentence_in = Variable(torch.LongTensor(sentence_in).cuda())
                        targets = torch.LongTensor(tags).cuda()
                        caps = Variable(torch.LongTensor(data['caps']).cuda())
                        feature = Variable(torch.FloatTensor(feature).cuda())
                        gaze_feature = Variable(torch.FloatTensor(gaze).cuda())
                        gaze_targets = Variable(torch.LongTensor(np.argmax(gaze, axis=1)).cuda())
                        # deps_targets = Variable(torch.LongTensor(np.argmax(deps, axis=1)).cuda())
                        shape_targets = Variable(torch.LongTensor(np.argmax(shapes, axis=1)).cuda())
                        pos_targets = Variable(torch.LongTensor(np.argmax(pos, axis=1)).cuda())
                    else:
                        sentence_in = Variable(torch.LongTensor(sentence_in))
                        targets = torch.LongTensor(tags)
                        caps = Variable(torch.LongTensor(data['caps']))
                        feature = Variable(torch.FloatTensor(feature))
                        gaze_feature = Variable(torch.FloatTensor(gaze))
                        gaze_targets = Variable(torch.LongTensor(np.argmax(gaze, axis=1)))
                        # deps_targets = Variable(torch.LongTensor(np.argmax(deps, axis=1)))
                        shape_targets = Variable(torch.LongTensor(np.argmax(shapes, axis=1)))
                        pos_targets = Variable(torch.LongTensor(np.argmax(pos, axis=1)))

#                     print("dep = ", dep.size())

                    neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets, chars2_mask, caps, chars2_length, d, feature, gaze_feature, gaze_targets, shape_targets, pos_targets)
                    loss += neg_log_likelihood.data[0] / len(data['words'])
                    neg_log_likelihood.backward()
                    torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
                    optimizer.step()

                print(phase + " : Loss = ", loss)
            if phase == "dev":
                model.eval()
                best_dev_F, new_dev_F, save_dev = evaluating(model, dataset["dev"], best_dev_F, features_dev, gaze_dev)
            if phase == "test":
                model.eval()
                best_test_F, new_test_F, save_test = evaluating(model, dataset["test"], best_test_F, features_test, gaze_test)
                if save_dev:
                    checkpoint_name = "checkpoints/checkpoint_" + str(new_test_F) + "_" + str(best_dev_F)+ "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pth"
                    print("[INFO] Save model at ", checkpoint_name)
                    torch.save(model, checkpoint_name)
                
model = Neural_CRF_AE(vocab_size=len(word_to_id),
                        tag_to_ix=tag_to_id,
                        embedding_dim=parameters['word_dim'],
                        hidden_dim=parameters['word_lstm_dim'],
                        use_gpu=use_gpu,
                        char_to_ix=char_to_id,
                        pre_word_embeds=word_embeds,
                        use_crf=parameters['crf'],
                        char_mode=parameters['char_mode'])

if use_gpu:
    model.cuda()

learning_rate = parameters["learning_rate"]
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
train_model(model, data, optimizer, step_lr_scheduler, num_epochs=parameters["epochs"])
