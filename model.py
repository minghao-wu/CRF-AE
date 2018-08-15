import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *
from config import parameters

START_TAG = '<START>'
STOP_TAG = '<STOP>'

def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Neural_CRF_AE(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_lstm_dim=parameters['char_lstm_dim'],
                 char_to_ix=None, pre_word_embeds=None, char_embedding_dim=parameters['char_dim'], use_gpu=False,
                 n_cap=None, cap_embedding_dim=None, use_crf=True, char_mode='CNN',
                 features_dim = parameters["features_dim"], gazetter_dim = parameters["gazetter_dim"], 
                 gazetteer_lambda = parameters["gazetteer_lambda"], pos_lambda = parameters["pos_lambda"],
                 wordshape_lambda = parameters["wordshape_lambda"]):
        super(Neural_CRF_AE, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.n_cap = n_cap
        self.cap_embedding_dim = cap_embedding_dim
        self.use_crf = use_crf
        self.tagset_size = len(tag_to_ix)
        self.out_channels = char_lstm_dim
        self.char_mode = char_mode


        
        self.hidden2gazetteer = nn.Linear(hidden_dim*2, gazetter_dim)
        # self.hidden2deps = nn.Linear(hidden_dim*2, 45)
        self.hidden2pos = nn.Linear(hidden_dim*2, 45)
        self.hidden2shape = nn.Linear(hidden_dim*2, 151)
        init_linear(self.hidden2gazetteer)
        # init_linear(self.hidden2deps)
        init_linear(self.hidden2pos)
        init_linear(self.hidden2shape)
        self.pos_lambda = pos_lambda
        self.wordshape_lambda = wordshape_lambda
        self.gazetteer_lambda = gazetteer_lambda

        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            init_embedding(self.char_embeds.weight)
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3, char_embedding_dim), padding=(2,0))

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_dim+self.out_channels+features_dim+gazetter_dim, hidden_dim, bidirectional=True)
        init_lstm(self.lstm)
        self.hw_trans = nn.Linear(self.out_channels, self.out_channels)
        self.hw_gate = nn.Linear(self.out_channels, self.out_channels)
        self.h2_h1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        init_linear(self.h2_h1)
        init_linear(self.hidden2tag)
        init_linear(self.hw_gate)
        init_linear(self.hw_trans)
        self.transitions = nn.Parameter(torch.zeros(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        r = torch.LongTensor(range(feats.size()[0]))
        if self.use_gpu:
            r = r.cuda()
            pad_start_tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.cuda.LongTensor([self.tag_to_ix[STOP_TAG]])])
        else:
            pad_start_tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
            pad_stop_tags = torch.cat([tags, torch.LongTensor([self.tag_to_ix[STOP_TAG]])])

        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def _get_lstm_features(self, sentence, chars2, caps, chars2_length, d, feature, gazetteer):
        chars_embeds = self.char_embeds(chars2).unsqueeze(1)
        chars_cnn_out3 = nn.functional.relu(self.char_cnn3(chars_embeds))
        chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,kernel_size=(chars_cnn_out3.size(2), 1)).view(chars_cnn_out3.size(0), self.out_channels)
        embeds = self.word_embeds(sentence)
        embeds = torch.cat((embeds, chars_embeds, feature, gazetteer), 1)
        embeds = embeds.unsqueeze(1)
        embeds = self.dropout(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim*2)
        lstm_out = self.dropout(lstm_out)
        gaze_feat = self.hidden2gazetteer(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        # deps_feats = self.hidden2deps(lstm_out)
        shape_feats = self.hidden2shape(lstm_out)
        pos_feats = self.hidden2pos(lstm_out)
        return lstm_feats, gaze_feat, shape_feats, pos_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            emit_score = feat.view(-1, 1)
            tag_var = forward_var + self.transitions + emit_score
            max_tag_var, _ = torch.max(tag_var, dim=1)
            tag_var = tag_var - max_tag_var.view(-1, 1)
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1) # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            viterbivars_t = Variable(torch.FloatTensor(viterbivars_t))
            if self.use_gpu:
                viterbivars_t = viterbivars_t.cuda()
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var.data[self.tag_to_ix[STOP_TAG]] = -10000.
        terminal_var.data[self.tag_to_ix[START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags, chars2, caps, chars2_length, d, feature, gazetteer, gaze_targets, shape_label, pos_label):
        # sentence, tags is a list of ints
        # features is a 2D tensor, len(sentence) * self.tagset_size
        feats, gaze, shapes, pos = self._get_lstm_features(sentence, chars2, caps, chars2_length, d, feature, gazetteer)

        lst = [3, 0.5, 3]
        if self.use_gpu:
            cls_weights = torch.cuda.FloatTensor(lst)
        else:
            cls_weights = torch.FloatTensor(lst)


        # hand_loss = self.feature_lambda * nn.functional.mse_loss(handcrafted, feature)
        gaze_loss = self.gazetteer_lambda * nn.functional.cross_entropy(gaze, gaze_targets, weight=cls_weights)
        # dep_loss = nn.functional.cross_entropy(deps, deps_label)
        shape_loss = nn.functional.cross_entropy(shapes, shape_label)
        pos_loss = nn.functional.cross_entropy(pos, pos_label)
        

        if self.use_crf:
            forward_score = self._forward_alg(feats)
            gold_score = self._score_sentence(feats, tags)
            return forward_score - gold_score + self.gazetteer_lambda*gaze_loss +  self.wordshape_lambda*shape_loss + self.pos_lambda*pos_loss
        else:
            tags = Variable(tags)
            scores = nn.functional.cross_entropy(feats, tags)
            return scores + self.gazetteer_lambda*gaze_loss +  self.wordshape_lambda*shape_loss + self.pos_lambda*pos_loss


    def forward(self, sentence, chars, caps, chars2_length, d, feature, gazetteer):
        feats, _, _, _ = self._get_lstm_features(sentence, chars, caps, chars2_length, d, feature, gazetteer)
        # viterbi to get tag_seq
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats)
        else:
            score, tag_seq = torch.max(feats, 1)
            tag_seq = list(tag_seq.cpu().data)

        return score, tag_seq
