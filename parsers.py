import time
import copy
import random
import argparse

import dynet_config

import constants as const
from arc_standard import ArcStandard
from conll18_ud_eval import load_conllu, load_conllu_file, evaluate
from data import ExperimentData, get_transition_index, get_label_index, \
    sents_to_conll


class RRTParser:
    def __init__(self):
        self.use_gpu = True
        if self.use_gpu:
            dynet_config.set_gpu()
        from networks import RRTNetwork
        self.name = 'rlt_model'
        self.param_path = 'rlt_model'
        self.mode = 'train'
        self.train_set_path = None
        self.dev_set_path = None
        self.test_set_path = None
        self.test_set_path2 = None
        self.test_conllu_gold_path = None
        self.dev_ud = None
        self.gold_ud = None
        self.conll_format = const.CONLL06
        self.wembpath = None
        self.tune_las = False

        self.use_bilstm_input = False
        self.in_lstm_size = 256
        self.in_lstm_count = 2
        self.hid_mlp_size = 256
        self.hid_mlp_count = 1
        self.encoding_type = 'forward'
        self.hid_lstm_size = 512
        self.hid_lstm_count = 2
        self.batch_size = 10

        self.stack_feats = 4
        self.buffer_feats = 4

        self.min_iter_for_test = 0
        self.iter_per_test = 500
        self.max_epochs = 30

        self.nn = RRTNetwork()
        self.exp_data = None
        self._session = None
        self._load_model_path = None

    @classmethod
    def from_args(cls, args):
        c = cls()
        c.name = args.model_path if args.model_path else c.name
        c.param_path = args.param_path if args.param_path else c.param_path
        c.mode = args.mode
        c.train_set_path = args.train_path
        c.dev_set_path = args.dev_path
        c.test_set_path = args.test_path
        c.test_set_path2 = args.test_path2
        c.test_conllu_gold_path = args.test_conllu_gold_path
        c.wembpath = args.word_emb_path
        c.conll_format = const.CONLL06 if args.data_format is 'conll06' else \
            const.CONLL09 if args.data_format is 'conll09' else const.CONLLU
        c.tune_las = args.tune == 'las'

        c.use_bilstm_input = args.use_bilstm_input
        c.in_lstm_size = args.input_bilstm_size
        c.in_lstm_count = args.input_bilstm_layers
        c.hid_mlp_size = args.hidden_layer_size
        c.hid_mlp_count = args.hidden_layers
        c.hid_lstm_size = args.encoding_lstm_size
        c.hid_lstm_count = args.encoding_lstm_layers
        c.batch_size = args.batch_size

        c.stack_feats = args.stack_feats
        c.buffer_feats = args.buffer_feats
        c.max_epochs = args.epochs
        c.iter_per_test = args.iter_per_test
        c.min_iter_for_test = args.min_iter_for_test

        from networks import RRTNetwork
        c.nn = RRTNetwork.from_args(args)
        if c.conll_format is const.CONLLU:
            c.dev_ud = load_conllu_file(c.dev_set_path)
            c.gold_ud = load_conllu_file(c.test_conllu_gold_path)
        return c

    def _save_params(self):
        param_file = open(self.name + '.params', 'w')
        param_file.write('--model_path ' + self.name + '\n')
        param_file.write('--param_path ' + self.param_path + '\n')
        param_file.write('--mode ' + self.mode + '\n')
        # param_file.write('--train_path ' + self.train_set_path + ' ')
        # param_file.write('--dev_path ' + self.dev_set_path + ' ')
        # param_file.write('--test_path ' + self.test_set_path + ' ')
        # if self.wembpath: param_file.write('--word_emb_path ' + self.wembpath + ' ')
        # param_file.write('--data_format ' + str(self.conll_format) + ' ')

        if self.use_bilstm_input: param_file.write('--use_bilstm_input\n')
        param_file.write('--input_bilstm_size ' + str(self.in_lstm_size) + '\n')
        param_file.write('--input_bilstm_layers ' + str(self.in_lstm_count) + '\n')
        param_file.write('--hidden_layer_size ' + str(self.hid_mlp_size) + '\n')
        param_file.write('--hidden_layers ' + str(self.hid_mlp_count) + '\n')
        param_file.write('--encoding_lstm_size ' + str(self.hid_lstm_size) + '\n')
        param_file.write('--encoding_lstm_layers ' + str(self.hid_lstm_count) + '\n')
        param_file.write('--batch_size ' + str(self.batch_size) + '\n')

        param_file.write('--stack_feats ' + str(self.stack_feats) + '\n')
        param_file.write('--buffer_feats ' + str(self.buffer_feats) + '\n')

        param_file.write('--min_iter_for_test ' + str(self.min_iter_for_test) + '\n')
        param_file.write('--iter_per_test ' + str(self.iter_per_test) + '\n')
        param_file.write('--epochs ' + str(self.max_epochs))
        param_file.close()

    def _build_graph(self):
        trans_out_size = 3  # SHIFT, LEFT/RIGHT_ARC
        label_out_size = len(self.exp_data.labels)
        self.nn.build_graph(self.exp_data.word_data.embeddings,
                            self.exp_data.pos_data.embeddings,
                            self.exp_data.dep_data.embeddings,
                            trans_out_size, label_out_size,
                            self.stack_feats, self.buffer_feats,
                            self.in_lstm_size, self.in_lstm_count,
                            self.hid_mlp_size, self.hid_mlp_count,
                            self.hid_lstm_size, self.hid_lstm_count)
        if self.mode == 'continue_train':
            print 'loading network model from', self.name + '.ckpt'
            self.nn.load(self.name + '.ckpt')

    def train(self):
        self.exp_data = ExperimentData.build_rrt_training_data(
            filename=self.train_set_path,
            conll_format=self.conll_format)
        emb_start = time.time()
        if self.wembpath is not None:
            self.exp_data.load_word_embeddings(self.wembpath)
        emb_end = time.time()
        print 'time to load embeddings:', str(emb_end-emb_start)
        if self.dev_set_path:
            self.exp_data.load_dev_sentences(self.dev_set_path,
                                             self.conll_format)
        if self.test_set_path:
            self.exp_data.load_test_sentences(self.test_set_path,
                                              self.conll_format)
        if self.test_set_path2:
            self.exp_data.load_test_sentences2(self.test_set_path2,
                                               self.conll_format)
        self._build_graph()
        words = [i for i in self.exp_data.input_data['words'] if i]
        pos = [i for i in self.exp_data.input_data['pos'] if i]
        trans = [i for i in self.exp_data.input_data['trans'] if i]
        deps = [i for i in self.exp_data.input_data['dep'] if i]
        outputs = [i for i in self.exp_data.output_data['trans'] if i]
        labels = [i for i in self.exp_data.output_data['dep'] if i]
        train_ids = range(0, len(words))
        print '# of training examples: ' + str(len(words))

        index = 0
        step = self.batch_size
        epochs = 0
        best_uas = 0
        best_las = 0
        best_iter = -1

        if self.mode == 'continue_train':
            print 'Initial dev. set score:'
            best_uas = uas = self.test(self.exp_data.dev_sentences)
            print 'Initial test set score:'
            self.test(self.exp_data.test_sentences)

        # self.test(self.exp_data.dev_sentences)
        for itera in range(0, 2000000):
            in_words = [words[i] for i in train_ids[index:index + step]]
            in_pos = [pos[i] for i in train_ids[index:index + step]]
            in_trans = [trans[i] for i in train_ids[index:index + step]]
            in_deps = [deps[i] for i in train_ids[index:index + step]]
            # in_extra = [extra[i] for i in train_ids[index:index + step]]
            out_trans = [outputs[i] for i in train_ids[index:index + step]]
            out_deps = [labels[i] for i in train_ids[index:index + step]]
            err, uacc, lacc = self.nn.train_batch(in_words,
                                                  in_pos,
                                                  in_trans,
                                                  in_deps,
                                                  out_trans,
                                                  out_deps)
            print '(' + str(epochs) + ',' + str(itera) + ') ' + str(err),
            print '(' + str(index) + ',' + str(index + step) + ')',
            print 'uas accuracy:' + str(uacc), 'las accuracy:', str(lacc)

            if index + step >= len(words):
                print 'end of epoch (' + str(epochs) + ')'
                random.shuffle(train_ids)
                index = 0
                epochs += 1
            else:
                index += step

            if (itera != 0 and (itera % self.iter_per_test == 0 and
                                itera >= self.min_iter_for_test)) or \
                    epochs >= self.max_epochs:
                print 'Dev. set scores:'
                test_start = time.time()
                uas, las = self.test(self.exp_data.dev_sentences, self.dev_ud)
                test_end = time.time()

                if (las > best_las and self.tune_las) or uas > best_uas:
                    best_uas = uas
                    best_las = las
                    best_iter = itera
                    if self.tune_las:
                        print 'Found new best las: ', best_las, ' @ ', best_iter
                    else:
                        print 'Found new best uas: ', best_uas, ' @ ', best_iter
                    # self.nn.save(self.name + '.model')
                    print 'Test set scores:'
                    self.test(self.exp_data.test_sentences, self.gold_ud)
                    if self.test_set_path2 is not None:
                        print 'Test set 2 scores:'
                        self.test(self.exp_data.test_sentences2, self.gold_ud)
                else:
                    print 'Current best uas: ', best_uas, ', best las:', best_las, ' @ ', best_iter

                print 'time taken for test ', str(test_end - test_start), 's'
            if epochs >= self.max_epochs:
                print 'Training finished'
                break

    def test(self, test_sentences, conllu_ud=None):
        splits = 64
        step = len(test_sentences) / splits
        start = 0
        uas = []
        las = []
        parsed_sents = []

        while start + step < len(test_sentences):
            u, l, p = self._test_part(test_sentences[start: start + step])
            uas.append(u)
            las.append(l)
            parsed_sents += p
            start += step
        u, l, p = self._test_part(test_sentences[start:])
        parsed_sents += p
        uas.append(u)
        las.append(l)
        lenu = len(uas)
        lenl = len(las)
        avgu = sum(uas) / lenu
        avgl = sum(las) / lenl
        if self.conll_format is const.CONLLU:
            res = sents_to_conll(parsed_sents)
            import cStringIO
            system_ud2 = load_conllu(cStringIO.StringIO(res))
            ev = evaluate(conllu_ud, system_ud2)
            avgu = ev['UAS'].f1
            avgl = ev['LAS'].f1
        print 'Test results:'
        print 'uas (no punctuation): ', str(avgu)
        print 'las (no punctuation): ', str(avgl)
        return avgu, avgl

    def _test_part(self, test_sentences):
        ua = la = total = ua_no_punc = la_no_punc = total_no_punc = 0.0

        parsers = []
        for sent in test_sentences:
            parsers.append(ArcStandard(copy.deepcopy(sent)))
        p_copy = parsers[:]

        words = [[t[s.format.FORM].lower() for t in s] for s in test_sentences]
        word_ids = [[self.exp_data.word_data.str_to_idx[w] if w in self.exp_data.word_data.str_to_idx
                     else self.exp_data.word_data.str_to_idx[const.unk_word] for w in word] for word in words]
        pos = [[t[s.format.PPOS] for t in s] for s in test_sentences]
        pos_ids = [[self.exp_data.pos_data.str_to_idx[p] for p in tags] for tags in pos]
        sentence_exps = None
        prev_trans = [-1] * len(parsers)
        prev_deps = [-1] * len(parsers)

        while parsers:
            trans_scores, label_scores, sentence_exps = self.nn.run(word_ids, pos_ids, sentence_exps, prev_trans, prev_deps)

            old_word_ids = word_ids
            old_pos_ids = pos_ids
            old_sentence_exps = sentence_exps
            word_ids = []
            pos_ids = []
            sentence_exps = {'buff': [], 'buffer_idx': [], 'stack': []}
            parsers_to_remove = []
            prev_trans = []
            prev_deps = []
            for i in range(0, len(parsers)):
                transition, _ = parsers[i].get_best_transition2(trans_scores[i], label_scores[i], self.exp_data.labels)
                is_done = parsers[i].apply(transition[0], transition[1])
                if is_done:
                    if self.conll_format is not const.CONLLU:
                        sent_eval = parsers[i].evaluate()
                        ua += sent_eval[0]
                        la += sent_eval[1]
                        total += sent_eval[2]
                        ua_no_punc += sent_eval[3]
                        la_no_punc += sent_eval[4]
                        total_no_punc += sent_eval[5]
                    parsers_to_remove.append(parsers[i])
                else:
                    word_ids.append(old_word_ids[i])
                    pos_ids.append(old_pos_ids[i])
                    sentence_exps['buff'].append(old_sentence_exps['buff'][i])
                    sentence_exps['buffer_idx'].append(old_sentence_exps['buffer_idx'][i])
                    sentence_exps['stack'].append(old_sentence_exps['stack'][i])
                    prev_trans.append(get_transition_index(parsers[i].transitions[-1]))
                    prev_deps.append(get_label_index(parsers[i].transitions[-1], self.exp_data.labels))

            for p in parsers_to_remove:
                if p.is_done():
                    parsers.remove(p)

        if self.conll_format is const.CONLLU:
            return 0, 0, [p._sentence for p in p_copy]

        uas = ua * 100.0 / total
        las = la * 100.0 / total
        uas_no_punc = ua_no_punc * 100.0 / total_no_punc
        las_no_punc = la_no_punc * 100.0 / total_no_punc

        return uas_no_punc, las_no_punc, [p._sentence for p in p_copy]


def set_args(parser):
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='\
            Where to save/load parser model. If the path given already has a model \
            of the same name and --mode is set to \
            \'train\' then the model will be overwritten.\
            '
    )
    parser.add_argument(
        '--param_path',
        type=str,
        default=None,
        help='\
            Where to save/load parser parameters. If the path given already has a \
            parameter file of the same name then that\
            parameter file will be used and will NOT be overwritten.\
            '
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'continue_train'],
        default='train',
        help='Whether to train a new model, continue training an existing model\
            , or test a trained model.'
    )
    parser.add_argument(
        '--tune',
        type=str,
        choices=['uas', 'las'],
        default='uas',
        help='Whether to pick the model that produces the best UAS or LAS on \
            the Dev. set.'
    )
    parser.add_argument(
        '--train_path',
        type=str,
        default=None,
        help='Path to training data. Must be given if --mode set to \'train\'.'
    )
    parser.add_argument(
        '--dev_path',
        type=str,
        default=None,
        help='Path to development data. Must be given if --mode set to \'train\'.'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='Path to test data. Must be given if --mode set to \'test\', \
            optional if --mode set to \'train\'.'
    )
    parser.add_argument(
        '--test_path2',
        type=str,
        default=None,
        help='Path to test data. Must be given if --mode set to \'test\', \
            optional if --mode set to \'train\'.'
    )
    parser.add_argument(
        '--test_conllu_gold_path',
        type=str,
        default=None,
        help='Must be given if --data_format is \'conllU\' or --test_path \
            will be ignored.'
    )
    parser.add_argument(
        '--word_emb_path',
        type=str,
        default=None,
        help='Path to pre-trained word embeddings. Optional if --mode set to \
            \'train\', if \'test\'.'
    )
    parser.add_argument(
        '--data_format',
        type=str,
        choices=['conll06', 'conll09', 'conllU'],
        default='conll06',
        help='Format of the data used for training/testing.'
    )

    parser.add_argument(
        '--use_bilstm_input',
        action='store_true',
        help='Use a bidirectional lstm layer to represent input sentences.'
    )
    parser.add_argument(
        '--input_bilstm_size',
        type=int,
        default=256,
        help='Size of bilstms used to represent words in an input sentence.'
    )
    parser.add_argument(
        '--input_bilstm_layers',
        type=int,
        default=2,
        help='Number of bilstm layers used to represent input sentences.'
    )
    parser.add_argument(
        '--hidden_layer_size',
        type=int,
        default=256,
        help='Size of fully connected hidden layers used.'
    )
    parser.add_argument(
        '--hidden_layers',
        type=int,
        default=1,
        help='Number of fully connected hidden layers used.'
    )
    parser.add_argument(
        '--encoding_lstm_size',
        type=int,
        default=256,
        help='Size of lstms/bilstms used to build the Recursive LSTM Tree.'
    )
    parser.add_argument(
        '--encoding_lstm_layers',
        type=int,
        default=2,
        help='Number of layers used to for the encoding mechanism.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='How many sentences to train on at a time.'
    )

    parser.add_argument(
        '--stack_feats',
        type=int,
        default=4,
        help='How many features to use from the top of the stack.'
    )
    parser.add_argument(
        '--buffer_feats',
        type=int,
        default=4,
        help='How many features to use from the front of the buffer.'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of passes through the training data before stopping.'
    )
    parser.add_argument(
        '--iter_per_test',
        type=int,
        default=500,
        help='How many iterations between each test of the model.'
    )
    parser.add_argument(
        '--min_iter_for_test',
        type=int,
        default=0,
        help='How many iterations to train at first without any testing.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    set_args(parser)
    args = parser.parse_args()
    if args.param_path:
        try:
            param_file = open(args.param_path)
            params = param_file.read().split()
            args = parser.parse_args(params)
            print 'Found existing parameter file at', args.param_path
        except IOError:
            pass
    print str(args)[10:-1].replace(', ', '\n')
    e = RRTParser.from_args(args)
    if args.mode == 'train' or args.mode == 'continue_train':
        e.train()
