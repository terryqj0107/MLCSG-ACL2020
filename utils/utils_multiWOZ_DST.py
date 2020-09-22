import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
import ast
from collections import Counter
from collections import OrderedDict
from embeddings import GloveEmbedding, KazumaCharEmbedding
from tqdm import tqdm
import os
import pickle
from random import shuffle

from .fix_label import *


if args['all_domains']:
    EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi", "hospital", "police"]
else:
    # 实验涉及的5个domains
    EXPERIMENT_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


# 构建词表
class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

    # 根据sent类型分别处理，加入词表
    def index_words(self, sent, type):
        if type == 'utter':
            for word in sent.split(" "):
                self.index_word(word)
        elif type == 'slot':
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
        elif type == 'belief':
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split(" "):
                    self.index_word(ss)
                for v in value.split(" "):
                    self.index_word(v)

    # 将词加入词表，更新词表
    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

# 构造合适的Dataset类，用于DataLoader喂数据
class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id, sequicity, mem_word2id):
        """Reads source and target sequences from txt files."""
        self.ID = data_info['ID']
        self.turn_domain = data_info['turn_domain']
        self.turn_id = data_info['turn_id']
        self.dialog_history = data_info['dialog_history']
        self.turn_belief = data_info['turn_belief']
        self.gating_label = data_info['gating_label']
        self.turn_uttr = data_info['turn_uttr']
        self.generate_y = data_info["generate_y"]
        self.sequicity = sequicity
        self.num_total_seqs = len(self.dialog_history)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.mem_word2id = mem_word2id
        self.lm_forward_trg_text = data_info['lm_forward_trg_text']
        self.lm_backward_trg_text = data_info['lm_backward_trg_text']
        self.system_response = data_info['system_response']

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        ID = self.ID[index]
        turn_id = self.turn_id[index]
        turn_belief = self.turn_belief[index]
        gating_label = self.gating_label[index]
        turn_uttr = self.turn_uttr[index]
        turn_domain = self.preprocess_domain(self.turn_domain[index])  # turn_domain的数字表示
        generate_y = self.generate_y[index]
        generate_y = self.preprocess_slot(generate_y, self.trg_word2id)  # generate_y文字转索引号
        context = self.dialog_history[index] 
        context = self.preprocess(context, self.src_word2id)  # 对话历史文字转索引号
        context_plain = self.dialog_history[index]
        lm_forward_trg_text = self.lm_forward_trg_text[index]
        lm_forward_trg_text_ids = self.preprocess(lm_forward_trg_text, self.src_word2id)
        lm_backward_trg_text = self.lm_backward_trg_text[index]
        lm_backward_trg_text_ids = self.preprocess(lm_backward_trg_text, self.src_word2id)
        system_response = self.system_response[index]
        system_response_ids = self.preprocess(system_response, self.src_word2id)

        item_info = {
            "ID": ID,
            "turn_id": turn_id,
            "turn_belief": turn_belief,
            "gating_label": gating_label,
            "context": context,   # 对话历史（id序列）
            "context_plain": context_plain,  # 对话历史（词序列）
            "turn_uttr_plain": turn_uttr,  # 上轮系统回复+本轮用户数据（词序列）
            "turn_domain": turn_domain,
            "generate_y": generate_y,  # generate_y标签的id序列
            "lm_forward_trg_text": lm_forward_trg_text,
            "lm_forward_trg_text_ids": lm_forward_trg_text_ids,
            "lm_backward_trg_text": lm_backward_trg_text,
            "lm_backward_trg_text_ids": lm_backward_trg_text_ids,
            "system_response": system_response,
            "system_response_ids": system_response_ids
            }
        return item_info

    def __len__(self):
        return self.num_total_seqs  # 数据集大小
    
    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_slot(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(v)
        # story = torch.Tensor(story)
        return story

    def preprocess_memory(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            d, s, v = value
            s = s.replace("book","").strip()
            # separate each word in value to different memory slot
            for wi, vw in enumerate(v.split()):
                idx = [word2idx[word] if word in word2idx else UNK_token for word in [d, s, "t{}".format(wi), vw]]
                story.append(idx)
        story = torch.Tensor(story)
        return story

    def preprocess_domain(self, turn_domain):
        domains = {"attraction":0, "restaurant":1, "taxi":2, "train":3, "hotel":4, "hospital":5, "bus":6, "police":7}
        return domains[turn_domain]


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)  # 取得所有序列中的最大长度
        padded_seqs = torch.ones(len(sequences), max_len).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_multi_response(sequences):
        '''
        merge from batch * nb_slot * slot_len to batch * nb_slot * max_slot_len
        '''
        lengths = []
        for bsz_seq in sequences:
            length = [len(v) for v in bsz_seq]
            lengths.append(length)
        max_len = max([max(l) for l in lengths])  # 取得槽值的最大长度 max_slot_len
        padded_seqs = []
        for bsz_seq in sequences:
            pad_seq = []
            for v in bsz_seq:
                v = v + [PAD_token] * (max_len-len(v))
                pad_seq.append(v)
            padded_seqs.append(pad_seq)
        padded_seqs = torch.tensor(padded_seqs)
        lengths = torch.tensor(lengths)
        return padded_seqs, lengths

    def merge_memory(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths) # avoid the empty belief state issue
        padded_seqs = torch.ones(len(sequences), max_len, 4).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if len(seq) != 0:
                padded_seqs[i,:end,:] = seq[:end]
        return padded_seqs, lengths
  
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context']), reverse=True) 
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences
    src_seqs, src_lengths = merge(item_info['context'])
    y_seqs, y_lengths = merge_multi_response(item_info["generate_y"])
    gating_label = torch.tensor(item_info["gating_label"])
    turn_domain = torch.tensor(item_info["turn_domain"])

    lm_forward_trg_text_ids, lm_forward_trg_text_ids_lengths = merge(item_info['lm_forward_trg_text_ids'])
    lm_backward_trg_text_ids, lm_backward_trg_text_ids_lengths = merge(item_info['lm_backward_trg_text_ids'])
    system_response_ids, system_response_ids_lengths = merge(item_info['system_response_ids'])
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        gating_label = gating_label.cuda()
        turn_domain = turn_domain.cuda()
        y_seqs = y_seqs.cuda()
        y_lengths = y_lengths.cuda()
        if args['LanguageModel']:
            lm_forward_trg_text_ids = lm_forward_trg_text_ids.cuda()
            lm_forward_trg_text_ids_lengths = torch.tensor(lm_forward_trg_text_ids_lengths).cuda()
            lm_backward_trg_text_ids = lm_backward_trg_text_ids.cuda()
            lm_backward_trg_text_ids_lengths = torch.tensor(lm_backward_trg_text_ids_lengths).cuda()
        if args['ResponseGeneration']:
            system_response_ids = system_response_ids.cuda()
            system_response_ids_lengths = torch.Tensor(system_response_ids_lengths).cuda()


    item_info["context"] = src_seqs
    item_info["context_len"] = src_lengths
    item_info["gating_label"] = gating_label
    item_info["turn_domain"] = turn_domain
    item_info["generate_y"] = y_seqs
    item_info["y_lengths"] = y_lengths

    item_info['lm_forward_trg_text_ids'] = lm_forward_trg_text_ids
    item_info['lm_forward_trg_text_ids_lengths'] = lm_forward_trg_text_ids_lengths
    item_info['lm_backward_trg_text_ids'] = lm_backward_trg_text_ids
    item_info['lm_backward_trg_text_ids_lengths'] = lm_backward_trg_text_ids_lengths

    item_info['system_response_ids'] = system_response_ids
    item_info['system_response_ids_lengths'] = system_response_ids_lengths
    return item_info


# 读取xxx_dials.json文件，更新词表
def read_langs(file_name, gating_dict, SLOTS, dataset, lang, mem_lang, sequicity, training, max_line = None):
    print(("Reading from {}".format(file_name)))
    data = []
    max_resp_len, max_value_len = 0, 0  # dialogue_history最大长度，value最大长度
    domain_counter = {}  # 统计每个domain出现的次数
    slot_temp = SLOTS
    with open(file_name) as f:
        dials = json.load(f)
        # create vocab first 
        for dial_dict in dials:
            if (args["all_vocab"] or dataset=="train") and training:
                for ti, turn in enumerate(dial_dict["dialogue"]):  # 遍历每段对话，将用户语句和系统语句中的词都加入词表
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
                    if args['separate_label']:
                        lang.index_word('[user]')
                        lang.index_word('[system]')
        # determine training data ratio, default is 100%
        if training and dataset == "train" and args["data_ratio"] != 100:
            random.Random(10).shuffle(dials)
            dials = dials[:int(len(dials)*0.01*args["data_ratio"])]
        
        cnt_lin = 1  # 统计行数
        for dial_dict in dials:
            dialog_history = ""
            last_belief_dict = {}
            # Filtering and counting domains
            for domain in dial_dict["domains"]:
                if domain not in EXPERIMENT_DOMAINS:  # 过滤domains,只考虑实验设置中的5个domains
                    continue
                if domain not in domain_counter.keys():  # 统计各个domain出现的次数
                    domain_counter[domain] = 0
                domain_counter[domain] += 1

            # Unseen domain setting
            if args["only_domain"] != "" and args["only_domain"] not in dial_dict["domains"]:
                continue
            if (args["except_domain"] != "" and dataset == "test" and args["except_domain"] not in dial_dict["domains"]) or \
               (args["except_domain"] != "" and dataset != "test" and [args["except_domain"]] == dial_dict["domains"]): 
                continue

            # Reading data
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_domain = turn["domain"]
                turn_id = turn["turn_idx"]

                if args['separate_label']:
                    if ti == 0 and turn["system_transcript"] == '':
                        turn_uttr = "[user] " + turn["transcript"]
                        dialog_history = turn_uttr
                    else:
                        turn_uttr = "[system] " + turn["system_transcript"] + " [user] " + turn["transcript"]
                        dialog_history += (" " + turn_uttr)
                    turn_uttr_strip = turn_uttr.strip()  # 拼接：[上轮系统回复;本轮用户语句]，并移除首尾空格
                else:
                    turn_uttr = turn["system_transcript"] + " ; " + turn["transcript"]
                    turn_uttr_strip = turn_uttr.strip()
                    dialog_history += (turn["system_transcript"] + " ; " + turn["transcript"] + " ; ")

                source_text = dialog_history.strip()  # 累计对话历史，并移除首尾空格

                lm_forward_trg_text = ' '.join(source_text.split()[1:]) + ' EOS'
                lm_backward_trg_text = 'SOS ' + ' '.join(source_text.split()[:-1])

                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, SLOTS)  # 对话状态字典，修正了一些对话状态标签错误
                # turn_belief_dict:  e.g. {'hotel-area': 'west', 'hotel-pricerange': 'cheap'}

                # Generate domain-dependent slot list
                slot_temp = SLOTS
                if dataset == "train" or dataset == "dev":
                    if args["except_domain"] != "":
                        # 当存在except_domain时，训练集和开发集：slot_temp保存除except_domain以外的slots，
                        # 验证集的turn_belief_dict也只保存除except_domain以外的对话状态
                        slot_temp = [k for k in SLOTS if args["except_domain"] not in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] not in k])
                    elif args["only_domain"] != "":
                        # 当存在only_domain时，slot_temp只保存only_domain中的slots，
                        # turn_belief_dict也只保存only_domain中的对话状态
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])
                else:
                    if args["except_domain"] != "":
                        # 当存在except_domain时，测试集：slot_temp保存except_domain中的slots，
                        # 测试集的turn_belief_dict保存except_domain中的对话状态
                        slot_temp = [k for k in SLOTS if args["except_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["except_domain"] in k])
                    elif args["only_domain"] != "":
                        # 当有only_domain时，测试集：也只保存only_domain的slots和对话状态
                        slot_temp = [k for k in SLOTS if args["only_domain"] in k]
                        turn_belief_dict = OrderedDict([(k, v) for k, v in turn_belief_dict.items() if args["only_domain"] in k])

                # 当前轮的对话状态列表，例如：hotel-pricerange-cheap
                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]

                if (args["all_vocab"] or dataset=="train") and training:  # 当训练模式时，对话状态字典的词更新到词表mem_lang
                    mem_lang.index_words(turn_belief_dict, 'belief')

                class_label, generate_y, slot_mask, gating_label = [], [], [], []
                start_ptr_label, end_ptr_label = [], []
                for slot in slot_temp:
                    if slot in turn_belief_dict.keys(): 
                        generate_y.append(turn_belief_dict[slot])
                        # generate_y：保存所有slots对应的值，当前提到的存具体value，没提到的存none
                        # gating_label：当前提到的slots，值为dontcare则存1，值为none则存2，否则存0；没提到的slots都存2
                        if turn_belief_dict[slot] == "dontcare":
                            gating_label.append(gating_dict["dontcare"])
                        elif turn_belief_dict[slot] == "none":
                            gating_label.append(gating_dict["none"])
                        else:
                            gating_label.append(gating_dict["ptr"])

                        if max_value_len < len(turn_belief_dict[slot]):  # max_value_len取最长的value长度
                            max_value_len = len(turn_belief_dict[slot])

                    else:
                        generate_y.append("none")
                        gating_label.append(gating_dict["none"])

                if ti < len(dial_dict["dialogue"])-1:
                    system_response = dial_dict["dialogue"][ti+1]["system_transcript"]
                else:
                    system_response = "none"

                data_detail = {
                    "ID": dial_dict["dialogue_idx"],
                    "domains": dial_dict["domains"],  # e.g.  ['taxi', 'hotel', 'attraction']
                    "turn_domain": turn_domain,       # e.g.  hotel
                    "turn_id": turn_id,
                    "dialog_history": source_text,  # 所有对话历史，拼接
                    "turn_belief": turn_belief_list,  # e.g.  ['hotel-area-west', 'hotel-pricerange-cheap']
                    "gating_label": gating_label,   # e.g.  [0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, ...... 2, 2, 2]
                    "turn_uttr": turn_uttr_strip,  # 上轮系统回复+当前用户语句
                    'generate_y': generate_y,  # e.g. ['cheap','none','none','none','none','none','west', ...... ,'none']
                    'lm_forward_trg_text': lm_forward_trg_text,  # source_text[1:] + EOS
                    'lm_backward_trg_text': lm_backward_trg_text,  # SOS + source_text[:-1]
                    'system_response': system_response
                    }
                data.append(data_detail)  # data: [ {'ID':'PMUL2082.json', 'domains': ..., 'turn_domain': ..., ....... }, { }, ... , { } ]
                
                if max_resp_len < len(source_text.split()):  # max_resp_len取最长的对话历史的长度
                    max_resp_len = len(source_text.split())
                
            cnt_lin += 1
            if(max_line and cnt_lin>=max_line):
                break

    # add t{} to the lang file
    if "t{}".format(max_value_len-1) not in mem_lang.word2index.keys() and training:
        for time_i in range(max_value_len):
            mem_lang.index_words("t{}".format(time_i), 'utter')

    print("domain_counter", domain_counter)
    return data, max_resp_len, slot_temp


# 得到合适的DataLoader实例
def get_seq(pairs, lang, mem_lang, batch_size, type, sequicity):  
    if(type and args['fisher_sample']>0):
        shuffle(pairs)
        pairs = pairs[:args['fisher_sample']]

    data_info = {}
    data_keys = pairs[0].keys()  # dict_keys(['ID', 'domains', 'turn_domain', 'turn_id', 'dialog_history', 'turn_belief', 'gating_label', 'turn_uttr', 'generate_y'])
    for k in data_keys:
        data_info[k] = []

    # data_info: 根据keys，把每个pair中的相应value归到一起
    # { 'ID':... , 'domains':... , 'turn_domain':... , ...   }
    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k])

    # 构造Dataset实例，用于DataLoader迭代数据
    dataset = Dataset(data_info, lang.word2index, lang.word2index, sequicity, mem_lang.word2index)

    # imbalance_sampler参数默认为0
    if args["imbalance_sampler"] and type:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  # shuffle=type,
                                                  collate_fn=collate_fn,
                                                  sampler=ImbalancedDatasetSampler(dataset))
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=type,
                                                  collate_fn=collate_fn)
    return data_loader


# 构造预训练的embedding
def dump_pretrained_emb(word2index, index2word, dump_path):
    print("Dumping pretrained embeddings...")
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    E = []
    for i in tqdm(range(len(word2index.keys()))):
        w = index2word[i]
        e = []
        for emb in embeddings:
            e += emb.emb(w, default='zero')
        E.append(e)
    with open(dump_path, 'wt') as f:
        json.dump(E, f)

# 获取ontology中的domain-slot对，只需要使用 domain-slot pairs
def get_slot_information(ontology):
    ontology_domains = dict([(k, v) for k, v in ontology.items() if k.split("-")[0] in EXPERIMENT_DOMAINS])
    SLOTS = [k.replace(" ", "").lower() if ("book" not in k) else k.lower() for k in ontology_domains.keys()]
    return SLOTS


# 数据预处理，加载数据
def prepare_data_seq(training, task="dst", sequicity=0, batch_size=100):
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size  # eval_batch默认为0，batch_size=32
    file_train = 'data/train_dials.json'
    file_dev = 'data/dev_dials.json'
    file_test = 'data/test_dials.json'
    # Create saving folder
    if args['path']:
        # folder_name = args['path'].rsplit('/', 2)[0] + '/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
        path_list = args['path'].split('/')
        folder_name = path_list[0] + '/' + path_list[1] + '/'
    else:
        folder_name = 'save/{}-'.format(args["decoder"])+args["addName"]+args['dataset']+str(args['task'])+'/'
    print("folder_name", folder_name)
    if not os.path.exists(folder_name): 
        os.makedirs(folder_name)
    # load domain-slot pairs from ontology
    ontology = json.load(open("data/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS = get_slot_information(ontology)  # 获取ontology中的domain-slot对，只需要使用 domain-slot pairs
    gating_dict = {"ptr": 0, "dontcare": 1, "none": 2}
    # Vocabulary
    lang, mem_lang = Lang(), Lang()  # 构建词表
    lang.index_words(ALL_SLOTS, 'slot')  # 将 domain，slot加入词表
    mem_lang.index_words(ALL_SLOTS, 'slot')
    lang_name = 'lang-all.pkl' if args["all_vocab"] else 'lang-train.pkl'  # all_vocab默认为1，lang-all.pkl表示全部词表
    mem_lang_name = 'mem-lang-all.pkl' if args["all_vocab"] else 'mem-lang-train.pkl'

    # 训练模式
    if training:
        # pair_train:  [ {'ID':'PMUL2082.json', 'domains': ..., 'turn_domain': ..., ....... }, { }, ... , { } ]
        pair_train, train_max_len, slot_train = \
            read_langs(file_train, gating_dict, ALL_SLOTS, "train", lang, mem_lang, sequicity, training)
        # train：训练集的数据迭代器
        train = get_seq(pair_train, lang, mem_lang, batch_size, True, sequicity)
        nb_train_vocab = lang.n_words  # 训练的词表大小

        pair_dev, dev_max_len, slot_dev = \
            read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)

        pair_test, test_max_len, slot_test = \
            read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

        if os.path.exists(folder_name+lang_name) and os.path.exists(folder_name+mem_lang_name):
            print("[Info] Loading saved lang files...")
            with open(folder_name+lang_name, 'rb') as handle: 
                lang = pickle.load(handle)
            with open(folder_name+mem_lang_name, 'rb') as handle: 
                mem_lang = pickle.load(handle)
        else:
            print("[Info] Dumping lang files...")
            with open(folder_name+lang_name, 'wb') as handle: 
                pickle.dump(lang, handle)
            with open(folder_name+mem_lang_name, 'wb') as handle: 
                pickle.dump(mem_lang, handle)
        emb_dump_path = 'data/emb{}.json'.format(len(lang.index2word))
        if not os.path.exists(emb_dump_path) and args["load_embedding"]:
            dump_pretrained_emb(lang.word2index, lang.index2word, emb_dump_path)
    # 测试模式
    else:
        with open(folder_name+lang_name, 'rb') as handle:
            lang = pickle.load(handle)
        with open(folder_name+mem_lang_name, 'rb') as handle:
            mem_lang = pickle.load(handle)

        pair_train, train_max_len, slot_train, train, nb_train_vocab = [], 0, {}, [], 0
        pair_dev, dev_max_len, slot_dev = read_langs(file_dev, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        dev = get_seq(pair_dev, lang, mem_lang, eval_batch, False, sequicity)
        pair_test, test_max_len, slot_test = read_langs(file_test, gating_dict, ALL_SLOTS, "test", lang, mem_lang, sequicity, training)
        test = get_seq(pair_test, lang, mem_lang, eval_batch, False, sequicity)

    test_4d = []
    if args['except_domain'] != "":
        pair_test_4d, _, _ = read_langs(file_test, gating_dict, ALL_SLOTS, "dev", lang, mem_lang, sequicity, training)
        test_4d = get_seq(pair_test_4d, lang, mem_lang, eval_batch, False, sequicity)

    max_word = max(train_max_len, dev_max_len, test_max_len) + 1

    print("Read %s pairs train" % len(pair_train))
    print("Read %s pairs dev" % len(pair_dev))
    print("Read %s pairs test" % len(pair_test))  
    print("Vocab_size: %s " % lang.n_words)  # 完整的词表大小，包括train+dev+test
    print("Vocab_size Training %s" % nb_train_vocab )
    print("Vocab_size Belief %s" % mem_lang.n_words )
    print("Max. length of dialog words for RNN: %s " % max_word)
    print("USE_CUDA={}".format(USE_CUDA))

    SLOTS_LIST = [ALL_SLOTS, slot_train, slot_dev, slot_test]
    print("[Train Set & Dev Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[2]))))
    print(SLOTS_LIST[2])
    print("[Test Set Slots]: Number is {} in total".format(str(len(SLOTS_LIST[3]))))
    print(SLOTS_LIST[3])
    LANG = [lang, mem_lang]
    return train, dev, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.turn_domain[idx]
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
