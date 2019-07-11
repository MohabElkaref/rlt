import time

from arc_standard import *
from conll18_ud_eval import *

import numpy as np


def build_dicts(sentences, conll_format, word_cutoff=1, case_sensitive=False, word_emb_dim=100, pos_emb_dim=50,
                dep_emb_dim=50):
    words = []
    pos = []
    deps = []
    word_counts = {}
    pos_counts = {}
    dep_counts = {}

    for s in sentences:
        if s and s != []:
            for token in s:
                form = token[conll_format.FORM].lower() if not case_sensitive else token[conll_format.FORM]
                if form in word_counts:
                    word_counts[form] += 1
                else:
                    words.append(form)
                    word_counts[form] = 1
                if token[conll_format.PPOS] in pos_counts:
                    pos_counts[token[conll_format.PPOS]] += 1
                else:
                    pos.append(token[conll_format.PPOS])
                    pos_counts[token[conll_format.PPOS]] = 1
                if token[conll_format.DEPREL] in dep_counts:
                    dep_counts[token[conll_format.DEPREL]] += 1
                elif token[conll_format.DEPREL] != '' and token[conll_format.DEPREL] != none_dep:
                    deps.append(token[conll_format.DEPREL])
                    dep_counts[token[conll_format.DEPREL]] = 1

    word_counts[none_word] = 0
    word_counts[unk_word] = 0
    words += [none_word, unk_word]
    pos_counts[none_pos] = 0
    pos += [none_pos]
    dep_counts[none_dep] = 0
    deps += [none_dep]

    reserved = [unk_word, none_word]
    removed = 0
    removed_words = []
    for k in word_counts.keys():
        if k not in reserved:
            if word_counts[k] <= word_cutoff:
                word_counts[unk_word] += word_counts[k]
                removed_words.append(k)
                words.remove(k)
                del word_counts[k]
                removed += 1
    print 'removed:', removed
    print 'unk count: ', word_counts[unk_word]

    word_idx = {}
    idx_word = {}
    for w in words:
        idx = len(word_idx)
        word_idx[w] = idx
        idx_word[idx] = w

    pos_idx = {}
    idx_pos = {}
    for p in pos:
        idx = len(pos_idx)
        pos_idx[p] = idx
        idx_pos[idx] = p

    dep_idx = {}
    idx_dep = {}
    for d in deps:
        idx = len(dep_idx)
        dep_idx[d] = idx
        idx_dep[idx] = d

    word_emb = np.random.normal(0, 0.01, (len(word_idx), word_emb_dim))
    pos_emb = np.random.normal(0, 0.01, (len(pos_idx), pos_emb_dim))
    dep_emb = np.random.normal(0, 0.01, (len(dep_idx), dep_emb_dim))

    return EmbeddingsDataSet(word_idx, idx_word, word_emb_dim, word_emb), \
           EmbeddingsDataSet(pos_idx, idx_pos, pos_emb_dim, pos_emb), \
           EmbeddingsDataSet(dep_idx, idx_dep, dep_emb_dim, dep_emb)


def read_conllu(filename, case_sensetive=False, add_root=True, root_first=True):
    ud_rep = load_conllu_file(filename)
    sentences = []
    for s in ud_rep.sentence_spans:
        words = ud_rep.words[s.start:s.end+1]
        tokens = [CONLLU.root_token[:]] + [w.columns for w in words]
        if not case_sensetive:
            for t in tokens:
                t[CONLLU.FORM] = t[CONLLU.FORM] #.lower()
        sentences.append(Sentence(tokens, CONLLU))
    return sentences


def read_conll(filename, conll_format, case_sensitive=False, add_root=True, root_first=True, test_flag=False):
    if conll_format is CONLLU and not test_flag:
        return read_conllu(filename, case_sensitive, add_root, root_first)
    lines = [[]]
    with open(filename) as f:
        for line in f:
            # if not case_sensitive:
            #     line = line.lower()
            if line.startswith('#'):
                continue
            line = line.strip()
            if line:
                line = line.split('\t')
                lines[-1].append(line)
            elif lines[-1] and lines[-1] != []:
                if add_root:
                    if root_first:
                        lines[-1].insert(0, conll_format.root_token[:])
                    else:
                        for token in lines[-1]:
                            if int(token[conll_format.HEAD]) == 0:
                                token[conll_format.HEAD] = str(len(lines[-1]))
                            else:
                                token[conll_format.HEAD] = int(token[conll_format.HEAD]) - 1
                        lines[-1].append(conll_format.root_token[:])
                lines[-1] = Sentence(lines[-1], conll_format)
                lines.append([])
    return lines[0:len(lines) - 1]


def sents_to_conll(sents):
    res = ''
    for s in sents:
        for t in s._all_tokens[1:]:
            token_string = ''
            for ts in t[:-1]:
                token_string += str(ts) + '\t'
            res += token_string + str(t[-1]) + '\n'
        res += '\n'
    return res


def read_embeddings(filename, words=None):
    emb_dict = {}
    with open(filename) as f:
        for line in f:
            line = line.strip().split()
            if line and len(line) > 2:
                if words and line[0] not in words:
                    continue
                for i in range(1, len(line)):
                    line[i] = float(line[i])
                emb_dict.update({line[0]: line[1:]})
    return emb_dict


class Sentence:
    INVALID_TOKEN_INDEX = -2

    def __init__(self, raw_tokens, conll_format):
        self._tokens = []
        self._all_tokens = []
        self.format = conll_format
        for t in raw_tokens:
            self._all_tokens.append(t)
            if '-' in t[self.format.INDEX] or '.' in t[self.format.INDEX]:
                continue
            self._tokens.append(t)
            self._tokens[-1][self.format.INDEX] = int(self._tokens[-1][self.format.INDEX])
            self._tokens[-1][self.format.HEAD] = Sentence.INVALID_TOKEN_INDEX \
                if self._tokens[-1][self.format.HEAD] is '_' else int(self._tokens[-1][self.format.HEAD])
            # self._tokens[-1][self.format.PHEAD] = Sentence.INVALID_TOKEN_INDEX
            # self._tokens[-1][self.format.PDEPREL] = ''

    def __getitem__(self, item):
        return self._tokens[item]

    def __len__(self):
        return len(self._tokens)

    def _get_all_children(self, node_id):
        if node_id >= len(self):
            return []
        indexes = []
        for i in range(0, len(self)):
            if self[i][self.format.HEAD] == node_id:
                indexes.append(i)
        return indexes

    def _get_node_string(self, indent, nodes, done_nodes):
        if not nodes:
            return ''
        result = ''
        for i in nodes:
            result += indent + '|\n'
            result += indent + '+ --- ' + self[i][self.format.PDEPREL] + '---- (' + str(i) + ') [' \
                      + self[i][self.format.PPOS] + '] ' + self[i][self.format.FORM] + '\n'
            done_nodes.append(i)
            result += self._get_node_string(indent + '|\t\t', self._get_all_children(i), done_nodes)
        return result

    def get_tree_string(self):
        return self._get_node_string('', self._get_all_children(-1), [])

    def get_leftmost_child(self, node_id, degree):
        index = -1
        if 0 < node_id < len(self):
            for i in range(0, node_id):
                if self[i][self.format.PHEAD] == node_id:
                    if degree == 1:
                        index = i
                        break
                    else:
                        degree -= 1
        return index

    def get_rightmost_child(self, node_id, degree):
        index = -1
        if 0 < node_id < len(self):
            for i in range(len(self) - 1, node_id, -1):
                if self[i][self.format.PHEAD] == node_id:
                    if degree == 1:
                        index = i
                        break
                    else:
                        degree -= 1
        return index

    def add_arc(self, dep_id, head_id, deprel):
        if dep_id > len(self) or head_id > len(self):
            return False
        self[dep_id][self.format.PHEAD] = head_id
        self[dep_id][self.format.PDEPREL] = deprel
        return True

    def evaluate(self):
        ua = la = total = 0
        ua_no_punc = la_no_punc = total_no_punc = 0

        for i in range(1, len(self)):
            punct = self[i][self.format.DEPREL].lower() == 'punct' \
                    or self[i][self.format.DEPREL].lower() == 'p'
            if self[i][self.format.PHEAD] == self[i][self.format.HEAD]:
                ua += 1
                if not punct:
                    ua_no_punc += 1
                if self[i][self.format.PDEPREL] == self[i][self.format.DEPREL]:
                    la += 1
                    if not punct:
                        la_no_punc += 1
            total += 1
            if not punct:
                total_no_punc += 1
        return [ua, la, total, ua_no_punc, la_no_punc, total_no_punc]

    def is_projective(self):
        for i in range(0, len(self)):
            for j in range(0, len(self)):
                if (i < self[i][self.format.HEAD] and i < j < self[i][self.format.HEAD] and
                        (self[j][self.format.HEAD] < i or self[j][self.format.HEAD] > self[i][self.format.HEAD])) or \
                        (i > self[i][self.format.HEAD] and i > j > self[i][self.format.HEAD] and
                             (self[j][self.format.HEAD] > i or self[j][self.format.HEAD] < self[i][self.format.HEAD])):
                    return False
        return True


class EmbeddingsDataSet:
    def __init__(self, str_to_idx, idx_to_str, dim=-1, embeddings=None):
        self.str_to_idx = str_to_idx
        self.idx_to_str = idx_to_str
        self.strings = self.idx_to_str.values()[:]
        self.dim = dim
        self.embeddings = embeddings


def get_output_index(transition, labels):
    if transition[2] == ArcStandard.SHIFT:
        return 0
    if transition[2] == ArcStandard.LEFT_ARC:
        return labels.index(transition[3]) + 1
    if transition[2] == ArcStandard.RIGHT_ARC:
        return len(labels) + labels.index(transition[3]) + 1
    return -1


def get_transition_index(transition):
    if transition[2] == ArcStandard.SHIFT:
        return 0
    if transition[2] == ArcStandard.LEFT_ARC:
        return 1
    if transition[2] == ArcStandard.RIGHT_ARC:
        return 2
    return -1


def get_label_index(transition, labels):
    if transition[2] == ArcStandard.SHIFT:
        return -1
    try:
        return labels.index(transition[3])
    except ValueError:
        return -1


class ExperimentData:
    def __init__(self):
        self.word_emb_dim = 100
        self.pos_emb_dim = 50
        self.dep_emb_dim = 50
        self.word_data = self.pos_data = self.dep_data = None

        self.labels = []
        self.features = []
        self.lex_count = 0
        self.pos_count = 0
        self.dep_count = 0

        self.input_data = []
        self.output_data = []
        self.train_sentences = []
        self.dev_sentences = []
        self.test_sentences = []
        self.test_sentences2 = []

    def load_dev_sentences(self, dev_filename, conll_format):
        self.dev_sentences = read_conll(dev_filename, conll_format, root_first=True, test_flag=True)

    def load_test_sentences(self, test_filename, conll_format):
        self.test_sentences = read_conll(test_filename, conll_format, root_first=True, test_flag=True)

    def load_test_sentences2(self, test_filename, conll_format):
        self.test_sentences2 = read_conll(test_filename, conll_format, root_first=True, test_flag=True)

    def load_word_embeddings(self, wemb_path):
        print 'loading word embeddings'
        wemb = read_embeddings(wemb_path, self.word_data.str_to_idx.keys())
        self.word_emb_dim = len(wemb.values()[0])
        self.word_data.dim = self.word_emb_dim
        word_emb = np.zeros((len(self.word_data.str_to_idx.keys()),
                             self.word_emb_dim))
        print 'length of wemb = ', len(wemb.keys())
        print 'length of vecs = ', len(word_emb)
        for w in wemb.keys():
            word_emb[self.word_data.str_to_idx[w]] = wemb[w]
        self.word_data.embeddings = word_emb

    @classmethod
    def build_training_data(cls, filename, conll_format, word_emb_dim=100, pos_emb_dim=50, dep_emb_dim=50,
                            wembpath=None, word_cutoff=1):
        data = cls()
        data.word_emb_dim = word_emb_dim
        data.pos_emb_dim = pos_emb_dim
        data.dep_emb_dim = dep_emb_dim
        start = time.time()
        all_sents = read_conll(filename, conll_format)
        data.train_sentences = []
        counter = 0
        proj = 0
        leng = len(all_sents)
        print 'length before filter: ', str(leng)
        for s in all_sents:
            if s.is_projective():
                data.train_sentences.append(s)
            else:
                proj += 1
            counter += 1
        data.word_data, data.pos_data, data.dep_data = build_dicts(data.train_sentences, conll_format, word_cutoff,
                                                                   word_emb_dim=data.word_emb_dim,
                                                                   pos_emb_dim=data.pos_emb_dim,
                                                                   dep_emb_dim=data.dep_emb_dim)
        data.labels = data.dep_data.idx_to_str.values()[:]
        data.labels.remove(none_dep)
        data.input_data = {'words': [], 'pos': [], 'trans': [], 'dep': []}
        data.output_data = []

        if wembpath:
            print 'reading word embeddings at', wembpath
            wemb = read_embeddings(wembpath, data.word_data.str_to_idx.keys())
            print 'length of word embeddings =', len(data.word_data.embeddings)
            print 'length of used word embeddings =', len(wemb.keys())
            for w in wemb.keys():
                data.word_data.embeddings[data.word_data.str_to_idx[w]] = wemb[w]

        total_sents = len(data.train_sentences)
        print 'total sents', str(total_sents)
        curr_sent_idx = 0
        frac = 0.05
        milestone = frac * total_sents
        print('parsing progress: ')
        for s in data.train_sentences:
            if not s:
                continue
            curr_sent_idx += 1
            # print curr_sent_idx
            if curr_sent_idx >= milestone:
                print str(frac * 100) + '% ..',
                frac += 0.05
                milestone = frac * total_sents
            parser = ArcStandard(s)
            # extract words/pos_tags for entire sentence here
            words = [t[conll_format.FORM] for t in s]
            word_ids = [data.word_data.str_to_idx[w] if w in data.word_data.str_to_idx
                        else data.word_data.str_to_idx[unk_word] for w in words]
            pos = [t[conll_format.PPOS] for t in s]
            pos_ids = [data.pos_data.str_to_idx[p] for p in pos]
            prev_trans = -1
            prev_dep = -1
            while not parser.is_done():
                data.input_data['words'].append(word_ids)
                data.input_data['pos'].append(pos_ids)
                data.input_data['trans'].append(prev_trans)
                data.input_data['dep'].append(prev_dep)

                parser.apply_gold()
                output_idx = get_output_index(parser.transitions[-1], data.labels)
                prev_trans = get_transition_index(parser.transitions[-1])
                prev_dep = get_label_index(parser.transitions[-1], data.labels)
                data.output_data.append(output_idx)
            data.input_data['words'].append([])
            data.input_data['pos'].append([])
            data.input_data['trans'].append(None)
            data.input_data['dep'].append(None)
            data.output_data.append(None)
        print 'done!'
        end = time.time()
        print 'time taken: ' + str(end - start) + ' s'
        return data

    @classmethod
    def build_hierarchical_training_data(cls, filename, conll_format, word_emb_dim=100, pos_emb_dim=50,
                                         dep_emb_dim=50, wembpath=None, word_cutoff=1):
        data = cls()
        data.word_emb_dim = word_emb_dim
        data.pos_emb_dim = pos_emb_dim
        data.dep_emb_dim = dep_emb_dim
        start = time.time()
        all_sents = read_conll(filename, conll_format)
        data.train_sentences = []
        counter = 0
        proj = 0
        leng = len(all_sents)
        print 'length before filter: ', str(leng)
        for s in all_sents:
            if s.is_projective():
                data.train_sentences.append(s)
            else:
                proj += 1
            counter += 1
        data.word_data, data.pos_data, data.dep_data = build_dicts(data.train_sentences, conll_format, word_cutoff,
                                                                   word_emb_dim=data.word_emb_dim,
                                                                   pos_emb_dim=data.pos_emb_dim,
                                                                   dep_emb_dim=data.dep_emb_dim)
        data.labels = data.dep_data.idx_to_str.values()[:]
        data.labels.remove(none_dep)
        data.input_data = {'words': [], 'pos': [], 'trans': [], 'dep': []}
        data.output_data = {'trans': [], 'label': []}

        if wembpath:
            print 'reading word embeddings at', wembpath
            wemb = read_embeddings(wembpath, data.word_data.str_to_idx.keys())
            print 'length of word embeddings =', len(data.word_data.embeddings)
            print 'length of used word embeddings =', len(wemb.keys())
            for w in wemb.keys():
                data.word_data.embeddings[data.word_data.str_to_idx[w]] = wemb[w]

        total_sents = len(data.train_sentences)
        print 'total sents', str(total_sents)
        curr_sent_idx = 0
        frac = 0.05
        milestone = frac * total_sents
        print('parsing progress: ')
        for s in data.train_sentences:
            if not s:
                continue
            curr_sent_idx += 1
            # print curr_sent_idx
            if curr_sent_idx >= milestone:
                print str(frac * 100) + '% ..',
                frac += 0.05
                milestone = frac * total_sents
            parser = ArcStandard(s)
            # extract words/pos_tags for entire sentence here
            words = [t[conll_format.FORM] for t in s]
            word_ids = [data.word_data.str_to_idx[w] if w in data.word_data.str_to_idx
                        else data.word_data.str_to_idx[unk_word] for w in words]
            pos = [t[conll_format.PPOS] for t in s]
            pos_ids = [data.pos_data.str_to_idx[p] for p in pos]
            prev_trans = -1
            prev_dep = -1
            while not parser.is_done():
                data.input_data['words'].append(word_ids)
                data.input_data['pos'].append(pos_ids)
                data.input_data['trans'].append(prev_trans)
                data.input_data['dep'].append(prev_dep)
                parser.apply_gold()
                prev_trans = trans_idx = get_transition_index(parser.transitions[-1])
                prev_dep = label_idx = get_label_index(parser.transitions[-1], data.labels)
                data.output_data['trans'].append(trans_idx)
                data.output_data['label'].append(label_idx)
            data.input_data['words'].append([])
            data.input_data['pos'].append([])
            data.input_data['trans'].append(None)
            data.input_data['dep'].append(None)
            data.output_data['trans'].append(None)
            data.output_data['label'].append(None)
        print 'done!'
        end = time.time()
        print 'time taken: ' + str(end - start) + ' s'
        return data

    def save_dicts(self, filename):
        print 'saving dictionaries to', filename, '...',
        with open(filename, 'a') as f:
            f.write(str(self.word_emb_dim) + '\n')
            f.write(str(self.pos_emb_dim) + '\n')
            f.write(str(self.dep_emb_dim) + '\n')
            f.write(str(len(self.word_data.str_to_idx)) + '\n')
            for w in self.word_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.word_data.str_to_idx[w]) + '\n')
            f.write(str(len(self.pos_data.str_to_idx)) + '\n')
            for w in self.pos_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.pos_data.str_to_idx[w]) + '\n')
            f.write(str(len(self.dep_data.str_to_idx)) + '\n')
            for w in self.dep_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.dep_data.str_to_idx[w]) + '\n')
            dep_count = len(self.labels)
            f.write(str(dep_count) + '\n')
            f.write(str(self.labels)[1:-1] + '\n')
        print 'done.'

    @staticmethod
    def load_dicts(filename):
        print 'loading dictionaries from', filename, '...'
        start = time.time()
        lines = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                lines.append(line)
                line = f.readline()

        data = ExperimentData()
        # load features
        data.word_emb_dim = int(lines[2])
        data.pos_emb_dim = int(lines[3])
        data.dep_emb_dim = int(lines[4])
        # load word dictionary
        print '\tloading word dictionary'
        wdict_length = int(lines[5])
        wdict_end = wdict_length + 6
        w_str_to_int = {}
        w_int_to_str = {}
        for i in range(6, wdict_end):
            entry = lines[i].strip().split()
            w_str_to_int[entry[0]] = int(entry[1])
            w_int_to_str[int(entry[1])] = entry[0]
        word_emb = np.random.normal(0, 0.01, (len(w_str_to_int.keys()), data.word_emb_dim))
        data.word_data = EmbeddingsDataSet(w_str_to_int, w_int_to_str, data.word_emb_dim, word_emb)
        # load pos dictionary
        print '\tloading pos dictionary'
        pdict_length = int(lines[wdict_end])
        pdict_end = wdict_end + pdict_length + 1
        p_str_to_int = {}
        p_int_to_str = {}
        for i in range(wdict_end + 1, pdict_end):
            entry = lines[i].strip().split()
            p_str_to_int[entry[0].lower()] = int(entry[1])
            p_int_to_str[int(entry[1])] = entry[0].lower()
        pos_emb = np.random.normal(0, 0.01, (len(p_str_to_int), data.pos_emb_dim))
        data.pos_data = EmbeddingsDataSet(p_str_to_int, p_int_to_str, data.pos_emb_dim, pos_emb)
        # load dep dictionary
        print '\tloading dep dictionary'
        ddict_length = int(lines[pdict_end])
        ddict_end = pdict_end + ddict_length + 1
        d_str_to_int = {}
        d_int_to_str = {}
        for i in range(pdict_end + 1, ddict_end):
            entry = lines[i].strip().split()
            d_str_to_int[entry[0].lower()] = int(entry[1])
            d_int_to_str[int(entry[1])] = entry[0].lower()
        dep_emb = np.random.normal(0, 0.01, (len(d_str_to_int), data.dep_emb_dim))
        data.dep_data = EmbeddingsDataSet(d_str_to_int, d_int_to_str, data.dep_emb_dim, dep_emb)
        # load labels in correct order
        print '\tloading labels'
        label_count = int(lines[ddict_end])
        data.labels = lines[ddict_end + 1].split(',')
        for i in range(0, len(data.labels)):
            data.labels[i] = data.labels[i].strip()[1:-1].lower()

        end = time.time()
        print 'done. time taken', str(end - start), '.'
        return data

    @classmethod
    def build_rrt_training_data(cls, filename, conll_format, word_cutoff=1):
        data = cls()
        start = time.time()
        all_sents = read_conll(filename, conll_format)
        data.train_sentences = []
        counter = 0
        proj = 0
        leng = len(all_sents)
        print 'length before removing non-projective sentences:', str(leng)
        for s in all_sents:
            if s.is_projective():
                data.train_sentences.append(s)
            else:
                proj += 1
            counter += 1
        data.word_data, data.pos_data, data.dep_data = \
            build_dicts(data.train_sentences, conll_format, word_cutoff)
        data.labels = data.dep_data.idx_to_str.values()[:]
        data.labels.remove(none_dep)
        data.input_data = {'words': [], 'pos': [], 'trans': [], 'dep': []}
        data.output_data = {'trans': [], 'dep': []}

        total_sents = len(data.train_sentences)
        print 'total projective sentences:', str(total_sents)
        curr_sent_idx = 0
        frac = 0.05
        milestone = frac * total_sents
        print('parsing progress: ')
        for s in data.train_sentences:
            if not s:
                continue
            curr_sent_idx += 1
            # print curr_sent_idx
            if curr_sent_idx >= milestone:
                print str(frac * 100) + '% '
                frac += 0.05
                milestone = frac * total_sents
            parser = ArcStandard(s)
            # extract words/pos_tags for entire sentence here
            words = [t[conll_format.FORM].lower() for t in s]
            word_ids = [data.word_data.str_to_idx[w]
                        if w in data.word_data.str_to_idx
                        else data.word_data.str_to_idx[unk_word] for w in words]
            pos = [t[conll_format.PPOS] for t in s]
            pos_ids = [data.pos_data.str_to_idx[p] for p in pos]

            data.input_data['words'].append(word_ids)
            data.input_data['pos'].append(pos_ids)
            prev_trans = -1
            prev_dep = -1
            trans = []
            dep = []
            output_trans = []
            output_deps = []
            while not parser.is_done():
                trans.append(prev_trans)
                dep.append(prev_dep)
                parser.apply_gold()
                prev_trans = trans_idx = get_transition_index(
                    parser.transitions[-1])
                prev_dep = label_idx = get_label_index(
                    parser.transitions[-1], data.labels)
                output_trans.append(trans_idx)
                output_deps.append(label_idx)
            data.input_data['trans'].append(trans)
            data.input_data['dep'].append(dep)
            data.output_data['trans'].append(output_trans)
            data.output_data['dep'].append(output_deps)
        end = time.time()
        print 'time taken: ' + str(end - start) + ' s'
        return data

    def write_rrt_to_file(self, filename):
        with open(filename, 'a') as f:
            f.write(str(len(self.word_data.str_to_idx)) + '\n')
            for w in self.word_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.word_data.str_to_idx[w]) + '\n')
            f.write(str(len(self.pos_data.str_to_idx)) + '\n')
            for w in self.pos_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.pos_data.str_to_idx[w]) + '\n')
            f.write(str(len(self.dep_data.str_to_idx)) + '\n')
            for w in self.dep_data.str_to_idx.keys():
                f.write(w + ' ' + str(self.dep_data.str_to_idx[w]) + '\n')
            dep_count = len(self.labels)
            f.write(str(dep_count) + '\n')
            f.write(str(self.labels)[1:-1] + '\n')
            f.write(str(len(self.input_data['words'])) + '\n')
            for i in range(0, len(self.input_data['words'])):
                f.write(str(self.input_data['words'][i])[1:-1] + '\n')
            for i in range(0, len(self.input_data['pos'])):
                f.write(str(self.input_data['pos'][i])[1:-1] + '\n')
            for i in range(0, len(self.input_data['trans'])):
                for j in range(0, len(self.input_data['trans'][i])):
                    f.write(str(self.input_data['trans'][i][j]) + '\n')
                f.write('\n')
            for i in range(0, len(self.input_data['dep'])):
                for j in range(0, len(self.input_data['dep'][i])):
                    f.write(str(self.input_data['dep'][i][j]) + '\n')
                f.write('\n')
            for i in range(0, len(self.output_data['trans'])):
                for j in range(0, len(self.output_data['trans'][i])):
                    f.write(str(self.output_data['trans'][i][j]) + '\n')
                f.write('\n')
            for i in range(0, len(self.output_data['dep'])):
                for j in range(0, len(self.output_data['dep'][i])):
                    f.write(str(self.output_data['dep'][i][j]) + '\n')
                f.write('\n')

    @staticmethod
    def load_rrt_from_file(filename, wembpath=None, wdim=0, pdim=0):
        start = time.time()
        lines = []
        with open(filename, 'r') as f:
            line = f.readline()
            while line:
                lines.append(line)
                line = f.readline()

        data = ExperimentData()
        # load word dictionary
        print 'loading word dictionary'
        wdict_length = int(lines[0])
        wdict_end = wdict_length + 1
        w_str_to_int = {}
        w_int_to_str = {}
        for i in range(1, wdict_end):
            entry = lines[i].strip().split()
            w_str_to_int[entry[0]] = int(entry[1])
            w_int_to_str[int(entry[1])] = entry[0]
        if wdim > 0:
            data.word_emb_dim = wdim
        if pdim > 0:
            data.pos_emb_dim = pdim
        data.word_data = EmbeddingsDataSet(w_str_to_int, w_int_to_str)
        if wembpath:
            data.load_word_embeddings(wembpath)
        # load pos dictionary
        print 'loading pos dictionary'
        pdict_length = int(lines[wdict_end])
        pdict_end = wdict_end + pdict_length + 1
        p_str_to_int = {}
        p_int_to_str = {}
        for i in range(wdict_end + 1, pdict_end):
            entry = lines[i].strip().split()
            p_str_to_int[entry[0].lower()] = int(entry[1])
            p_int_to_str[int(entry[1])] = entry[0].lower()
        data.pos_data = EmbeddingsDataSet(p_str_to_int, p_int_to_str)
        # load dep dictionary
        print 'loading dep dictionary'
        ddict_length = int(lines[pdict_end])
        ddict_end = pdict_end + ddict_length + 1
        d_str_to_int = {}
        d_int_to_str = {}
        for i in range(pdict_end + 1, ddict_end):
            entry = lines[i].strip().split()
            d_str_to_int[entry[0].lower()] = int(entry[1])
            d_int_to_str[int(entry[1])] = entry[0].lower()
        data.dep_data = EmbeddingsDataSet(d_str_to_int, d_int_to_str)
        # load labels in correct order
        print 'loading labels'
        label_count = int(lines[ddict_end])
        data.labels = lines[ddict_end + 1].split(',')
        for i in range(0, len(data.labels)):
            data.labels[i] = data.labels[i].strip()[1:-1].lower()
        data_length = int(lines[ddict_end + 2])
        # load input data
        print 'loading input data'
        data.input_data = {'words': [], 'pos': [], 'trans': [], 'dep': []}
        input_start = ddict_end + 3
        input_end = input_start + data_length
        print 'loading word ids'
        for i in range(input_start, input_end):
            entry = lines[i].split(',')
            for j in range(0, len(entry)):
                if entry[j] == '\n':
                    entry.remove(entry[j])
                else:
                    entry[j] = int(entry[j].strip())
            data.input_data['words'].append(entry)
        input_start = input_end
        input_end = input_start + data_length
        print 'loading pos ids'
        for i in range(input_start, input_end):
            entry = lines[i].split(',')
            for j in range(0, len(entry)):
                if entry[j] == '\n':
                    entry.remove(entry[j])
                else:
                    entry[j] = int(entry[j].strip())
            data.input_data['pos'].append(entry)
        input_start = input_end
        input_end = input_start + data_length
        print 'loading transitions'
        for i in range(0, data_length):
            trans = []
            while lines[input_start].strip():
                trans.append(int(lines[input_start]))
                input_start += 1
            input_start += 1
            data.input_data['trans'].append(trans)
        input_start = input_end
        input_end = input_start + data_length
        print 'loading dependency labels'
        for i in range(0, data_length):
            trans = []
            while lines[input_start].strip():
                trans.append(int(lines[input_start]))
                input_start += 1
            input_start += 1
            data.input_data['dep'].append(trans)
        print 'loading output transitions'
        output_start = input_start
        data.output_data = {'trans': [], 'dep': []}
        for i in range(0, data_length):
            entry = []
            while lines[output_start].strip():
                entry.append(int(lines[output_start]))
                output_start += 1
            output_start += 1
            data.output_data['trans'].append(entry)
        print 'loading output dependency labels'
        for i in range(0, data_length):
            entry = []
            while lines[output_start].strip():
                entry.append(int(lines[output_start]))
                output_start += 1
            output_start += 1
            data.output_data['dep'].append(entry)

        end = time.time()
        print 'time taken', str(end - start)
        return data
