import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np

# import matplotlib.pyplot as plt
# import seaborn  as sns
# import nltk
import os
import json
# import pandas as pd
import copy

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
import pprint

class TRADE(nn.Module):
    def __init__(self, hidden_size, lang, path, task, lr, dropout, slots, gating_dict, nb_train_vocab=0):
        super(TRADE, self).__init__()
        self.name = "TRADE"
        self.task = task
        self.hidden_size = hidden_size    
        self.lang = lang[0]
        self.mem_lang = lang[1] 
        self.lr = lr 
        self.dropout = dropout
        self.slots = slots[0]
        self.slot_temp = slots[2]
        self.gating_dict = gating_dict
        self.nb_gate = len(gating_dict)
        self.cross_entorpy = nn.CrossEntropyLoss()
        self.encoder = EncoderRNN(self.lang.n_words, hidden_size, self.dropout)
        self.decoder = Generator(self.lang, self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout, self.slots, self.nb_gate)

        if args['LanguageModel']:
            self.language_model = LanguageModel(self.encoder.embedding, self.lang.n_words, hidden_size, self.dropout)
            if USE_CUDA:
                self.language_model.cuda()
            self.lm_cross_entropy = nn.CrossEntropyLoss(ignore_index=1)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path)+'/enc.th')
                trained_decoder = torch.load(str(path)+'/dec.th')
                if args['LanguageModel']:
                    trained_language_model = torch.load(str(path) + '/language_model.th')
                    self.language_model.load_state_dict(trained_language_model.state_dict())
            else:
                print("MODEL {} LOADED".format(str(path)))
                trained_encoder = torch.load(str(path)+'/enc.th', lambda storage, loc: storage)
                trained_decoder = torch.load(str(path)+'/dec.th', lambda storage, loc: storage)
                if args['LanguageModel']:
                    trained_language_model = torch.load(str(path) + '/language_model.th', lambda storage, loc: storage)
                    self.language_model.load_state_dict(trained_language_model.state_dict())
            
            self.encoder.load_state_dict(trained_encoder.state_dict())
            self.decoder.load_state_dict(trained_decoder.state_dict())

        # Initialize optimizers and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1, min_lr=0.0001, verbose=True)
        
        self.reset()
        if USE_CUDA:
            self.encoder.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_ptr = self.loss_ptr / self.print_every
        print_loss_gate = self.loss_gate / self.print_every
        print_loss_class = self.loss_class / self.print_every
        # print_loss_domain = self.loss_domain / self.print_every
        self.print_every += 1

        if args['LanguageModel']:
            return 'L:{:.2f},LP:{:.2f},LG:{:.2f},LM_loss:{:.2f}'.format(
                    print_loss_avg, print_loss_ptr, print_loss_gate, self.lm_loss)
        else:
            return 'L:{:.2f},LP:{:.2f},LG:{:.2f}'.format(print_loss_avg,print_loss_ptr,print_loss_gate)
    
    def save_model(self, dec_type):
        directory = 'save/TRADE-'+args["addName"]+args['dataset']+str(self.task)+'/'+'HDD'+str(self.hidden_size)\
                    + 'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+str(dec_type)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.decoder, directory + '/dec.th')
        if args['LanguageModel']:
            torch.save(self.language_model, directory + '/language_model.th')
    
    def reset(self):
        self.loss, self.print_every, self.loss_ptr, self.loss_gate, self.loss_class = 0, 1, 0, 0, 0

    def train_batch(self, data, clip, slot_temp, reset=0):
        if reset: self.reset()
        # # Zero gradients of both optimizers
        # self.optimizer.zero_grad()
        
        # Encode and Decode
        use_teacher_forcing = random.random() < args["teacher_forcing_ratio"]  # teacher_forcing_ratio默认为0.5

        if args['LanguageModel']:
            # [B, max_len, V], [B, max_len, V]
            lm_forward_outputs, lm_backward_outputs, \
            all_point_outputs, gates, words_point_out, words_class_out, encoded_outputs, encoded_hidden = \
                self.encode_and_decode(data, use_teacher_forcing, slot_temp, True)
            lm_forward_outputs = lm_forward_outputs.permute(0, 2, 1)  # [B, V, max_len]
            lm_backward_outputs = lm_backward_outputs.permute(0, 2, 1)
            lm_forward_trg_text_ids = data['lm_forward_trg_text_ids']  # [B, max_len]
            lm_backward_trg_text_ids = data['lm_backward_trg_text_ids']
            lm_forward_loss = self.lm_cross_entropy(lm_forward_outputs, lm_forward_trg_text_ids)
            lm_backward_loss = self.lm_cross_entropy(lm_backward_outputs, lm_backward_trg_text_ids)
            self.lm_loss = lm_forward_loss + lm_backward_loss
        else:
            # [slot_num, B, max_res_len, V], [slot_num, B, gates], list[slot_num * max_len * B],[],[B, max_len, H],[1, B, H]
            all_point_outputs, gates, words_point_out, words_class_out, encoded_outputs, encoded_hidden = \
                self.encode_and_decode(data, use_teacher_forcing, slot_temp, True)

        loss_ptr = masked_cross_entropy_for_value(
            all_point_outputs.transpose(0, 1).contiguous(),
            data["generate_y"].contiguous(),  # [:,:len(self.point_slots)].contiguous(),
            data["y_lengths"])  # [:,:len(self.point_slots)])
        loss_gate = self.cross_entorpy(gates.transpose(0, 1).contiguous().view(-1, gates.size(-1)),
                                       data["gating_label"].contiguous().view(-1))

        if args["use_gate"]:
            loss = loss_ptr + loss_gate
        else:
            loss = loss_ptr

        if args['LanguageModel']:
            loss = loss + args['beta'] * self.lm_loss

        self.loss_grad = loss
        self.loss_ptr_to_bp = loss_ptr
        
        # Update parameters with optimizers
        self.loss += loss.data
        self.loss_ptr += loss_ptr.item()
        self.loss_gate += loss_gate.item()


    def optimize(self, clip, batch_id):
        self.loss_grad.backward()
        if args['delay_update']:
            if (batch_id + 1) % args['delay_update'] == 0:
                for p in self.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.data.mul_(1 / float(args['delay_update']))
                clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)  # 梯度裁剪，防止梯度爆炸
                self.optimizer.step()
                # Zero gradients of both optimizers
                self.optimizer.zero_grad()

        else:
            clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), clip)  # 梯度裁剪，防止梯度爆炸
            self.optimizer.step()
            # Zero gradients of both optimizers
            self.optimizer.zero_grad()

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def encode_and_decode(self, data, use_teacher_forcing, slot_temp, training):
        # Build unknown mask for memory to encourage generalization
        # 参数unk_mask默认为False，随机掩码为了提高泛化能力
        if args['unk_mask'] and self.decoder.training:
            story_size = data['context'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask = rand_mask * bi_mask
            rand_mask = torch.Tensor(rand_mask)
            if USE_CUDA: 
                rand_mask = rand_mask.cuda()
            story = data['context'] * rand_mask.long()
        else:
            story = data['context']  # 对话历史, [B, max_context_len]

        if args['LanguageModel']:
            # [B, max_len, H], [B, max_len, V], [B, max_len, V], [2, B, H]
            lm_outputs, lm_forward_outputs, lm_backward_outputs, lm_hidden = \
                self.language_model(story.transpose(0, 1), data['context_len'])
            encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'], lm_outputs, lm_hidden)
        else:
            # Encode dialog history
            # encoded_outputs: [B, max_len, H]  , encoded_hidden: [1, B, H]
            encoded_outputs, encoded_hidden = self.encoder(story.transpose(0, 1), data['context_len'], None, None)

        # Get the words that can be copy from the memory
        batch_size = len(data['context_len'])
        self.copy_list = data['context_plain']
        # data['generate_y']: [B, slot_num, max_value_len(max_res_len)]
        max_res_len = data['generate_y'].size(2) if self.encoder.training else 10  # 生成对话状态value的最大长度
        # [slot_num, B, max_res_len, V] , [slot_num, B, gates] , list[] slot_num * max_len * B , []
        all_point_outputs, all_gate_outputs, words_point_out, words_class_out = self.decoder.forward(batch_size,
            encoded_hidden, encoded_outputs, data['context_len'], story, max_res_len, data['generate_y'],
            use_teacher_forcing, slot_temp, training)


        if args['LanguageModel']:
            return lm_forward_outputs, lm_backward_outputs, \
                       all_point_outputs, all_gate_outputs, words_point_out, words_class_out, encoded_outputs, encoded_hidden
        else:
            return all_point_outputs, all_gate_outputs, words_point_out, words_class_out, encoded_outputs, encoded_hidden

    def evaluate(self, dev, matric_best, slot_temp, early_stop=None):
        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)  
        print("STARTING EVALUATION")
        all_prediction = {}
        inverse_unpoint_slot = dict([(v, k) for k, v in self.gating_dict.items()])
        pbar = tqdm(enumerate(dev),total=len(dev))
        for j, data_dev in pbar: 
            # Encode and Decode
            batch_size = len(data_dev['context_len'])
            if args['LanguageModel']:
                # [B, max_len, V], [B, max_len, V]
                lm_forward_outputs, lm_backward_outputs,\
                _, gates, words, class_words, encoded_outputs, encoded_hidden = \
                    self.encode_and_decode(data_dev, False, slot_temp, False)
            else:
                _, gates, words, class_words, encoded_outputs, encoded_hidden = \
                    self.encode_and_decode(data_dev, False, slot_temp, False)

            for bi in range(batch_size):
                if data_dev["ID"][bi] not in all_prediction.keys():
                    all_prediction[data_dev["ID"][bi]] = {}
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]] = {"turn_belief":data_dev["turn_belief"][bi]}
                predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
                gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

                # pointer-generator results
                if args["use_gate"]:
                    for si, sg in enumerate(gate):
                        if sg==self.gating_dict["none"]:
                            continue
                        elif sg==self.gating_dict["ptr"]:
                            pred = np.transpose(words[si])[bi]
                            st = []
                            for e in pred:
                                if e== 'EOS': break
                                else: st.append(e)
                            st = " ".join(st)
                            if st == "none":
                                continue
                            else:
                                predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+inverse_unpoint_slot[sg.item()])
                else:
                    for si, _ in enumerate(gate):
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e == 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(slot_temp[si]+"-"+str(st))

                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr
                all_prediction[data_dev["ID"][bi]][data_dev["turn_id"][bi]]["Dialogue History"] = data_dev["context_plain"][bi]

                # 当预测错误时，打印结果样例进行对比
                # if set(data_dev["turn_belief"][bi]) != set(predict_belief_bsz_ptr) and args["genSample"]:
                # if args["genSample"]:
                #     print('******************************************************')
                #     print('Dialogue_id: \n', data_dev["ID"][bi], "\n")
                #     print('Turn_id: \n', data_dev["turn_id"][bi], "\n")
                #     print("Dialogue History: \n", data_dev["context_plain"][bi], "\n")
                #     print("True:  ", set(data_dev["turn_belief"][bi]))
                #     print("Pred:  ", set(predict_belief_bsz_ptr), "\n")

        if args["genSample"]:
            json.dump(all_prediction, open("all_prediction_{}_52.04_with_utterances.json".format(self.name), 'w'), indent=4)

        joint_acc_score_ptr, F1_score_ptr, turn_acc_score_ptr = self.evaluate_metrics(all_prediction, "pred_bs_ptr", slot_temp)

        evaluation_metrics = {"Joint Acc":joint_acc_score_ptr, "Turn Acc":turn_acc_score_ptr, "Joint F1":F1_score_ptr}
        print(evaluation_metrics)

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        joint_acc_score = joint_acc_score_ptr # (joint_acc_score_ptr + joint_acc_score_class)/2
        F1_score = F1_score_ptr

        if (early_stop == 'F1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("MODEL SAVED")  
            return F1_score
        else:
            if (joint_acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(joint_acc_score))
                print("MODEL SAVED")
            return joint_acc_score

    def evaluate_metrics(self, all_prediction, from_which, slot_temp):
        total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    joint_acc += 1
                total += 1

                # Compute prediction slot accuracy
                temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
                turn_acc += temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                F1_pred += temp_f1
                F1_count += count

        joint_acc_score = joint_acc / float(total) if total!=0 else 0
        turn_acc_score = turn_acc / float(total) if total!=0 else 0
        F1_score = F1_pred / float(F1_count) if F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, n_layers=1):
        super(EncoderRNN, self).__init__()      
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size  
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # self.domain_W = nn.Linear(hidden_size, nb_domain)

        # 加载预训练word embedding
        if args["load_embedding"]:
            with open(os.path.join("data/", 'emb{}.json'.format(vocab_size))) as f:
                E = json.load(f)
            new = self.embedding.weight.data.new
            self.embedding.weight.data.copy_(new(E))
            self.embedding.weight.requires_grad = True
            print("Encoder embedding requires_grad", self.embedding.weight.requires_grad)

        if args["fix_embedding"]:
            self.embedding.weight.requires_grad = False

    # 构造初始化的隐状态[2, B, H]
    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, input_seqs, input_lengths, lm_outputs, lm_hidden):
        embedded = self.embedding(input_seqs)  # [max_len, B] → [max_len, B, E]

        if args['LanguageModel']:
            lm_outputs = lm_outputs.transpose(0, 1)
            embedded = embedded + lm_outputs  # [max_len, B, H]

        embedded = self.dropout_layer(embedded)  # dropout

        if args['lm_last_hidden']:
            hidden = lm_hidden  # language model 的 last hidden 作为初始化encoder hidden [2, B, H]
        else:
            hidden = self.get_state(input_seqs.size(1))  # 获取初始化隐状态，初始化为全0。 [2, B, H]
        # input_lengths: list[ , , ... , ]  length=B
        if input_lengths:
            # 将一个填充后的变长序列压紧。
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)  # outputs:PackedSequence(data=,batch_sizes=),  hidden: [2, B, H]
        if input_lengths:
            # 把压紧的序列再填充回来。
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)     # [max_len, B, 2*H]
        hidden = hidden[0] + hidden[1]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # [max_len, B, H]
        return outputs.transpose(0, 1), hidden.unsqueeze(0)


class LanguageModel(nn.Module):
    def __init__(self, shared_emb, vocab_size, hidden_size, dropout, n_layers=1):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = shared_emb
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    # 构造初始化的隐状态[2, B, H]
    def get_state(self, bsz):
        """Get cell states and hidden states."""
        if USE_CUDA:
            return Variable(torch.zeros(2, bsz, self.hidden_size)).cuda()
        else:
            return Variable(torch.zeros(2, bsz, self.hidden_size))

    def lm_attend_vocab(self, seq, cond):
        '''
        :param seq:  [V, H]
        :param cond:  [max_len, B, H]
        :return:  [max_len, B, V]
        '''
        scores_ = cond.matmul(seq.transpose(1, 0))  # [max_len, B, V]
        scores = F.softmax(scores_, dim=1)  # [max_len, B, V]
        return scores

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)  # [max_len, B] → [max_len, B, E]
        embedded = self.dropout_layer(embedded)  # dropout
        hidden = self.get_state(input_seqs.size(1))  # 获取初始化隐状态，初始化为全0。 [2, B, H]
        # input_lengths: list[ , , ... , ]  length=B
        if input_lengths:
            # 将一个填充后的变长序列压紧。
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=False)
        outputs, hidden = self.gru(embedded, hidden)  # outputs:PackedSequence(data=,batch_sizes=),  hidden: [2, B, H]
        if input_lengths:
            # 把压紧的序列再填充回来。
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=False)     # [max_len, B, 2*H]
        forward_outputs = outputs[:, :, :self.hidden_size]  # [max_len, B, H]
        backward_outputs = outputs[:, :, self.hidden_size:]  # [max_len, B, H]
        lm_outputs = forward_outputs + backward_outputs  # [max_len, B, H]
        forward_outputs = self.lm_attend_vocab(self.embedding.weight, forward_outputs)  # [max_len, B, V]
        backward_outputs = self.lm_attend_vocab(self.embedding.weight, backward_outputs)  # [max_len, B, V]
        return lm_outputs.transpose(0, 1), forward_outputs.transpose(0, 1), backward_outputs.transpose(0, 1), hidden


class Generator(nn.Module):
    def __init__(self, lang, shared_emb, vocab_size, hidden_size, dropout, slots, nb_gate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.lang = lang
        self.embedding = shared_emb 
        self.dropout_layer = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, dropout=dropout)
        self.nb_gate = nb_gate
        self.hidden_size = hidden_size
        self.W_ratio = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.slots = slots
        self.W_gate = nn.Linear(hidden_size, nb_gate)  # 分类器（三分类：ptr, dontcare, none）

        # Create independent slot embeddings
        self.slot_w2i = {}
        for slot in self.slots:
            if slot.split("-")[0] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[0]] = len(self.slot_w2i)
            if slot.split("-")[1] not in self.slot_w2i.keys():
                self.slot_w2i[slot.split("-")[1]] = len(self.slot_w2i)
        self.Slot_emb = nn.Embedding(len(self.slot_w2i), hidden_size)
        self.Slot_emb.weight.data.normal_(0, 0.1)

    def forward(self, batch_size, encoded_hidden, encoded_outputs, encoded_lens, story, max_res_len, target_batches, use_teacher_forcing, slot_temp, training):
        # [slot_num, B, max_res_len, V]  生成每个slot的value的概率分布
        all_point_outputs = torch.zeros(len(slot_temp), batch_size, max_res_len, self.vocab_size)
        # [slot_num, B, gates]  输出的三分类概率分布
        all_gate_outputs = torch.zeros(len(slot_temp), batch_size, self.nb_gate)
        if USE_CUDA: 
            all_point_outputs = all_point_outputs.cuda()
            all_gate_outputs = all_gate_outputs.cuda()

            # Get the slot embedding  例如：把hotel-pricerange的hotel和pricerange分别做embedding，然后相加成一个
        slot_emb_dict = {}  # slot与其embedding的映射字典
        for i, slot in enumerate(slot_temp):
            # Domain embbeding
            if slot.split("-")[0] in self.slot_w2i.keys():
                domain_w2idx = [self.slot_w2i[slot.split("-")[0]]]
                domain_w2idx = torch.tensor(domain_w2idx)
                if USE_CUDA: domain_w2idx = domain_w2idx.cuda()
                domain_emb = self.Slot_emb(domain_w2idx)   # [1, E]
            # Slot embbeding
            if slot.split("-")[1] in self.slot_w2i.keys():
                slot_w2idx = [self.slot_w2i[slot.split("-")[1]]]
                slot_w2idx = torch.tensor(slot_w2idx)
                if USE_CUDA: slot_w2idx = slot_w2idx.cuda()
                slot_emb = self.Slot_emb(slot_w2idx)   # [1, E]

            # Combine two embeddings as one query
            combined_emb = domain_emb + slot_emb  # [1, E]
            slot_emb_dict[slot] = combined_emb
            slot_emb_exp = combined_emb.expand_as(encoded_hidden)  # 用于并行计算，加速decoding。可不用，避免OOM。
            if i == 0:
                slot_emb_arr = slot_emb_exp.clone()
            else:
                slot_emb_arr = torch.cat((slot_emb_arr, slot_emb_exp), dim=0)

        # Compute pointer-generator output, decoding each (domain, slot) one-by-one
        words_point_out = []
        counter = 0
        # 依次遍历每一个slot，解码得到相应的value
        for slot in slot_temp:
            hidden = encoded_hidden   # 使用encoded_hidden作为解码状态的初始化，  [1, B, H]
            words = []

            slot_emb = slot_emb_dict[slot]  # [1, E]

            decoder_input = self.dropout_layer(slot_emb).expand(batch_size, self.hidden_size)  # [B, H]
            # 逐词生成当前slot对应的value
            for wi in range(max_res_len):
                # decoder_input.expand_as(hidden): [1, B ,H]
                dec_state, hidden = self.gru(decoder_input.expand_as(hidden), hidden)   # [1, B, H]
                '''
                   Attention Mechanism  当前解码状态与encoded_outputs计算attention。
                   参数 encoded_outputs: [B, max_len, H]  ,  hidden.squeeze(0): [B, H] ,  encoded_lens: list[]
                   返回值 context_vec: [B, H] ,  logits: [B, max_len] , prob: [B, max_len]
                '''
                context_vec, logits, prob = self.attend(encoded_outputs, hidden.squeeze(0), encoded_lens)
                if wi == 0:  # 使用解码端第一个context_vec去做分类（三分类：ptr, dontcare, none）
                    all_gate_outputs[counter] = self.W_gate(context_vec)  # [slot_num, B, gates]

                if args['modifyGen']:
                    p_vocab = self.modify_attend_vocab(self.embedding.weight, hidden.squeeze(0), context_vec)  # [B, V]
                else:
                    p_vocab = self.attend_vocab(self.embedding.weight, hidden.squeeze(0))  # vocab上的分布, [B, V]

                # p_gen取决于当前t时刻的decode state，context vector，decoder input
                p_gen_vec = torch.cat([dec_state.squeeze(0), context_vec, decoder_input], -1)  # [B, 3*H]
                vocab_pointer_switches = self.sigmoid(self.W_ratio(p_gen_vec))  # 从vocab中生成的概率值p_gen
                p_context_ptr = torch.zeros(p_vocab.size())  # [B, V]
                if USE_CUDA: p_context_ptr = p_context_ptr.cuda()
                p_context_ptr.scatter_add_(1, story, prob)  # 将context范围内copy概率分布映射到词表范围  [B, V]
                # 当前t时刻生成词的最终概率分布, [B, V]
                final_p_vocab = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr + \
                                vocab_pointer_switches.expand_as(p_context_ptr) * p_vocab
                pred_word = torch.argmax(final_p_vocab, dim=1)  # 选取概率值最高的词作为当前t时刻的生成词  [B]
                words.append([self.lang.index2word[w_idx.item()] for w_idx in pred_word])
                all_point_outputs[counter, :, wi, :] = final_p_vocab  # [slot_num, B, max_res_len, V]
                if use_teacher_forcing:
                    decoder_input = self.embedding(target_batches[:, counter, wi])  # Chosen word is next input
                else:
                    decoder_input = self.embedding(pred_word)
                if USE_CUDA: decoder_input = decoder_input.cuda()
            counter += 1
            words_point_out.append(words)
        
        return all_point_outputs, all_gate_outputs, words_point_out, []

    def attend(self, seq, cond, lens):
        """
        attend over the sequences `seq` using the condition `cond`.
        参数 seq: [B, max_len, H]  ,  cond: [B, H] ,  lens: list[]
        返回值 context: [B, H] ,  scores_: [B, max_len] , scores: [B, max_len]
        """
        scores_ = cond.unsqueeze(1).expand_as(seq).mul(seq).sum(2)  # 点积attention， 得到attention scores: [B, max_len]
        max_len = max(lens)   # 取得当前batch中序列的最大长度
        for i, l in enumerate(lens):
            if l < max_len:  # 填充符的位置得分置为无穷小
                scores_.data[i, l:] = -np.inf
        scores = F.softmax(scores_, dim=1)   # attention distribution : [B, max_len]
        # 按scores概率分布，计算encoder的seq的加权求和，得到当前t时刻的注意力向量，即上下文向量context vector
        context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
        return context, scores_, scores

    def attend_vocab(self, seq, cond):
        '''
        :param seq:  [V, H]
        :param cond:  [B, H]
        :return:  [B, V]
        '''
        scores_ = cond.matmul(seq.transpose(1, 0))  # [B, V]
        scores = F.softmax(scores_, dim=1)  # [B, V]
        return scores

    def modify_attend_vocab(self, embs, d_s, c_v):
        '''
        :param embs:  [V, H]
        :param d_s:  [B, H]
        :param v_v:  [B, H]
        :return:  [B, V]
        '''
        fused_state = torch.cat((d_s, c_v), dim=1)  # [B, 2*H]
        fused_state = self.generate_linear(fused_state)  # [B, H]
        scores_ = fused_state.matmul(embs.transpose(1, 0))  # [B, V]
        scores = F.softmax(scores_, dim=1)  # [B, V]
        return scores


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))