import numpy as np
import dynet as dy


class RRTNetwork:
    def __init__(self):
        self._initialised = False
        self.pc = dy.ParameterCollection()
        self.params = {}
        self.in_bilstm = None
        self.in_bilstm_dp = 0.3
        self.hid_lstm = None
        self.hid_lstm_dp = 0.3
        self.hid_mlp_dp = 0.3
        self.optimiser = None
        self.joint_in = False
        self.stack_feats = 4
        self.buffer_feats = 4
        self.use_bilstm_in = False
        self.rrt = None
        self.use_labels = False

    @classmethod
    def from_args(cls, args):
        c = cls()
        c.use_bilstm_in = args.use_bilstm_input
        c.stack_feats = args.stack_feats
        c.buffer_feats = args.buffer_feats
        return c

    def build_graph(self,
                    word_embs, pos_embs, dep_embs,
                    trans_out_size, label_out_size,
                    stack_feat_count=4, buffer_feat_count=4,
                    in_bilstm_size=256, in_lstm_count=2,
                    hid_mlp_size=256, hid_mlp_count=2,
                    hid_lstm_size=256, hid_lstm_count=2):
        self.stack_feats = stack_feat_count
        self.buffer_feats = buffer_feat_count
        self.params['word_embs'] = self.pc.lookup_parameters_from_numpy(word_embs)
        self.params['rand_word_embs'] = self.pc.lookup_parameters_from_numpy(
            np.zeros_like(word_embs))
        self.params['pos_embs'] = self.pc.lookup_parameters_from_numpy(pos_embs)
        self.params['dep_embs'] = self.pc.lookup_parameters_from_numpy(dep_embs)
        input_dim = self.params['word_embs'].shape()[1] + \
                    self.params['pos_embs'].shape()[1]
        dep_size = self.params['dep_embs'].shape()[1] \
            if self.use_labels and dep_embs is not None else 0

        if self.use_bilstm_in:
            self.in_bilstm = dy.BiRNNBuilder(in_lstm_count, input_dim,
                                             2 * in_bilstm_size, self.pc,
                                             dy.VanillaLSTMBuilder)
            self.hid_lstm = dy.VanillaLSTMBuilder(hid_lstm_count,
                                                  2*in_bilstm_size +
                                                  dep_size + hid_lstm_size,
                                                  hid_lstm_size, self.pc)
        else:
            self.hid_lstm = dy.VanillaLSTMBuilder(hid_lstm_count,
                                                  input_dim+dep_size +
                                                  hid_lstm_size,
                                                  hid_lstm_size, self.pc)

        self.params['empty_feats'] = [self.pc.add_parameters(hid_lstm_size)
                                      for _ in range(self.stack_feats)]
        self.params['empty_buffer_feats'] = [
            self.pc.add_parameters(hid_lstm_size)
            for _ in range(self.buffer_feats)]
        self.params['init_dep'] = self.pc.add_parameters(hid_lstm_size +
                                                         dep_size)
        self.params['end_dep'] = self.pc.add_parameters(hid_lstm_size +
                                                        dep_size)
        self.params['trans_mlp_w'] = [
            self.pc.add_parameters((hid_mlp_size,
                                    (self.stack_feats + self.buffer_feats) *
                                    hid_lstm_size
                                    if i == 0 else hid_mlp_size))
            for i in range(hid_mlp_count)]
        self.params['label_mlp_w'] = [
            self.pc.add_parameters((hid_mlp_size,
                                    (self.stack_feats + self.buffer_feats) *
                                    hid_lstm_size
                                    if i == 0 else hid_mlp_size))
            for i in range(hid_mlp_count)]
        self.rrt = RLT

        self.params['trans_mlp_b'] = [self.pc.add_parameters((hid_mlp_size))
                                      for _ in range(hid_mlp_count)]
        self.params['trans_out_w'] = self.pc.add_parameters((trans_out_size,
                                                             hid_mlp_size))

        self.params['label_mlp_b'] = [self.pc.add_parameters((hid_mlp_size))
                                      for _ in range(hid_mlp_count)]
        self.params['label_out_w'] = self.pc.add_parameters((label_out_size,
                                                             hid_mlp_size))

        self.optimiser = dy.AdamTrainer(self.pc)
        self._initialised = True

    def train_batch(self, word_ids, pos_ids, transitions, deps, outputs, labels):
        assert len(word_ids) == len(pos_ids) == len(transitions) == len(outputs), \
            'Lengths of input examples are not equal'
        dy.renew_cg()
        loss = []
        correct_uas = 0.
        correct_las = 0.
        total_uas = 0.
        total_las = 0.
        word_embs = self.params['word_embs']
        rand_word_embs = self.params['rand_word_embs']
        pos_embs = self.params['pos_embs']
        dep_embs = self.params['dep_embs']
        empty_feats = self.params['empty_feats']
        empty_buffer_feats = self.params['empty_buffer_feats']
        init_dep = dy.parameter(self.params['init_dep'])
        end_dep = dy.parameter(self.params['end_dep'])
        if self.use_bilstm_in:
            self.in_bilstm.set_dropout(self.in_bilstm_dp)
        self.hid_lstm.set_dropout(self.hid_lstm_dp)
        trans_w = [dy.parameter(mw) for mw in self.params['trans_mlp_w']]
        trans_b = [dy.parameter(mb) for mb in self.params['trans_mlp_b']]
        trans_o = dy.parameter(self.params['trans_out_w'])
        label_w = [dy.parameter(mw) for mw in self.params['label_mlp_w']]
        label_b = [dy.parameter(mb) for mb in self.params['label_mlp_b']]
        label_o = dy.parameter(self.params['label_out_w'])
        for wids, pids, transition, dep, outs, lbls in \
                zip(word_ids, pos_ids, transitions, deps, outputs, labels):
            wembs = [dy.lookup(word_embs, id) for id in wids]
            rwembs = [dy.lookup(rand_word_embs, id) for id in wids]
            swembs = [r + dy.nobackprop(w) for r, w in zip(rwembs, wembs)]
            pembs = [dy.lookup(pos_embs, id) for id in pids]
            embs = [dy.concatenate([we, pe]) for we, pe in zip(swembs, pembs)]
            bilstm_in = embs
            if not self.use_bilstm_in:
                sentence = embs
            else:
                sentence = self.in_bilstm.transduce(bilstm_in)
            buffer_idx = 1
            buff = [self.rrt(s, init_dep, self.hid_lstm, end_dep)
                    for s in sentence]
            stack = [buff[0]]
            for trans, d, out, lbl in zip(transition, dep, outs, lbls):
                if trans == 0:  # SHIFT
                    stack.append(buff[buffer_idx])
                    buffer_idx = buffer_idx + 1 if buffer_idx < len(buff)-1 else -1
                elif trans == 1:  # LEFT-ARC
                    stack[-1].add_left_dependent(stack[-2], dy.lookup(dep_embs, d))
                    stack.pop(-2)
                elif trans == 2:  # RIGHT-ARC
                    stack[-2].add_right_dependent(stack[-1], dy.lookup(dep_embs, d))
                    stack.pop()
                stack_states = [stack[-i-1].get_state() if len(stack) > i
                                else dy.parameter(empty_feats[i])
                                for i in range(self.stack_feats)]
                buffer_states = [buff[i + buffer_idx].get_state()
                                 if len(buff) > (i + buffer_idx)
                                 else dy.parameter(empty_buffer_feats[i])
                                 for i in range(self.buffer_feats)]

                input_layer = dy.concatenate(stack_states + buffer_states)
                input_layer = dy.dropout(input_layer, self.hid_mlp_dp)
                trans_mlp = input_layer
                for i in range(len(trans_w)):
                    trans_mlp = dy.dropout(dy.rectify(trans_w[i] * trans_mlp +
                                                      trans_b[i]),
                                           self.hid_mlp_dp)
                trans_scores = trans_o * trans_mlp
                loss.append(dy.pickneglogsoftmax(trans_scores, out))
                correct_uas += np.argmax(trans_scores.vec_value()) == out
                total_uas += 1

                if lbl > -1:
                    label_mlp = input_layer
                    for i in range(len(label_w)):
                        label_mlp = dy.dropout(dy.rectify(label_w[i] *
                                                          label_mlp+label_b[i]),
                                               self.hid_mlp_dp)
                    label_scores = label_o * label_mlp
                    loss.append(dy.pickneglogsoftmax(label_scores, lbl))
                    correct_las += np.argmax(label_scores.vec_value()) == lbl
                    total_las += 1

        loss = dy.average(loss)
        loss_value = loss.value()
        loss.backward()
        self.optimiser.update()

        return loss_value, correct_uas / total_uas, correct_las / total_las

    def run(self, sentence_wids, sentence_pids, sentence_exps, trans_ids, dep_ids):
        using_ids = False
        if sentence_exps is None and sentence_wids and sentence_pids:
            assert len(sentence_wids) == len(sentence_pids), \
                'Lengths of sentence_wids and sentence_pids does not match'
            assert len(sentence_wids) == len(trans_ids), \
                'Length of sentence_wids/pids and trans_ids must match'
            assert len(dep_ids) == len(trans_ids), \
                'Length of dep_ids and trans_ids must match'
            using_ids = True
        assert using_ids or sentence_exps, \
            'Either precomputed sentence_exps, or sentence_wids/pids must be given'

        if using_ids:
            dy.renew_cg()
        word_embs = self.params['word_embs']
        rand_word_embs = self.params['rand_word_embs']
        pos_embs = self.params['pos_embs']
        dep_embs = self.params['dep_embs']
        empty_feats = self.params['empty_feats']
        empty_buffer_feats = self.params['empty_buffer_feats']
        init_dep = dy.parameter(self.params['init_dep'])
        end_dep = dy.parameter(self.params['end_dep'])
        if self.use_bilstm_in:
            self.in_bilstm.disable_dropout()
        trans_w = [dy.parameter(mw) for mw in self.params['trans_mlp_w']]
        trans_b = [dy.parameter(mb) for mb in self.params['trans_mlp_b']]
        trans_o = dy.parameter(self.params['trans_out_w'])
        label_w = [dy.parameter(mw) for mw in self.params['label_mlp_w']]
        label_b = [dy.parameter(mb) for mb in self.params['label_mlp_b']]
        label_o = dy.parameter(self.params['label_out_w'])
        if using_ids:
            sentence_exps = {'buff': [], 'buffer_idx': [], 'stack': []}
            for wids, pids in zip(sentence_wids, sentence_pids):
                assert len(wids) == len(pids), \
                    'Length of wids and pids does not match for a sentence'
                wembs = [dy.lookup(word_embs, id) for id in wids]
                rwembs = [dy.lookup(rand_word_embs, id) for id in wids]
                swembs = [r + w for r, w in zip(rwembs, wembs)]
                pembs = [dy.lookup(pos_embs, id) for id in pids]
                embs = [dy.concatenate([we, pe]) for we, pe in zip(swembs, pembs)]
                bilstm_in = embs
                if not self.use_bilstm_in:
                    sentence = embs
                else:
                    sentence = self.in_bilstm.transduce(bilstm_in)

                buff = [self.rrt(s, init_dep, self.hid_lstm, end_dep)
                        for s in sentence]
                sentence_exps['buff'].append(buff)
                sentence_exps['buffer_idx'].append(1)
                sentence_exps['stack'].append([buff[0]])

        assert len(sentence_exps) == 3 and sentence_exps['buff'] and \
               sentence_exps['stack'], 'Malformed sentence_exps'
        assert len(sentence_exps['buff']) == len(sentence_exps['stack']) == \
               len(trans_ids), 'Length of sentence_exps and trans_ids must match'
        trans_score_vals = []
        label_score_vals = []
        for sent_id, (trans, dep) in enumerate(zip(trans_ids, dep_ids)):
            buff = sentence_exps['buff'][sent_id]
            stack = sentence_exps['stack'][sent_id]
            buff_idx = sentence_exps['buffer_idx'][sent_id]
            if trans == 0:  # SHIFT
                stack.append(buff[buff_idx])
                sentence_exps['buffer_idx'][sent_id] = buff_idx = buff_idx + 1 \
                    if buff_idx < len(buff) - 1 else -1
            elif trans == 1:  # LEFT-ARC
                stack[-1].add_left_dependent(stack[-2], dy.lookup(dep_embs, dep))
                stack.pop(-2)
            elif trans == 2:  # RIGHT-ARC
                stack[-2].add_right_dependent(stack[-1], dy.lookup(dep_embs, dep))
                stack.pop()

            stack_states = [stack[-i-1].get_state() if len(stack) > i
                            else dy.parameter(empty_feats[i])
                            for i in range(self.stack_feats)]
            buffer_states = [buff[i + buff_idx].get_state()
                             if len(buff) > (i + buff_idx)
                             else dy.parameter(empty_buffer_feats[i])
                             for i in range(self.buffer_feats)]
            input_layer = dy.concatenate(stack_states + buffer_states)
            trans_mlp = input_layer
            for i in range(len(trans_w)):
                trans_mlp = dy.rectify(trans_w[i] * trans_mlp + trans_b[i])
            trans_scores = trans_o * trans_mlp
            trans_score_vals.append(trans_scores.vec_value())

            label_mlp = input_layer
            for i in range(len(label_w)):
                label_mlp = dy.rectify(label_w[i] * label_mlp + label_b[i])
            label_scores = label_o * label_mlp
            label_score_vals.append(label_scores.vec_value())

        return trans_score_vals, label_score_vals, sentence_exps

    def save(self, save_path):
        assert self._initialised, 'Network parameters are not initialised. Please run build_graph() first.'
        self.pc.save(fname=save_path)

    def load(self, load_path):
        assert self._initialised, 'Network parameters are not initialised. Please run build_graph() first.'
        self.pc.populate(fname=load_path)


class RLT:
    def __init__(self, head, start_emb, rnn, end_emb=None):
        self._head = head
        self._start_emb = start_emb
        self._end_emb = start_emb if end_emb is None else end_emb
        self._head_tag = [dy.concatenate([self._head, self._start_emb])]
        self._end_tag = [dy.concatenate([self._head, self._end_emb])]
        self._dependents = []
        self._labels = []
        self.rnn = rnn
        self._init_state = self.rnn.initial_state().add_input(
            dy.concatenate([head, start_emb]))
        self._state = self._init_state
        self._output = None
        self._add_in_order = True  # all new dependents are added in the order in which they appear in the sentence
        self.use_end_tag = False  # add a trailing end tag to the child sequence when returning output
        self.use_labels = False

    def get_state(self):
        if not self.use_end_tag:
            return self._state.output()
        if self._output is None:
            self._output = self._state.transduce(self._end_tag)[-1]
        return self._output

    def add_right_dependent(self, dependent, label=None):
        self._dependents.append(dependent)
        if self.use_labels:
            self._labels.append(label)
            self._state = self._state.add_input(dy.concatenate([self._head, label, dependent.get_state()]))
        else:
            self._state = self._state.add_input(dy.concatenate([self._head, dependent.get_state()]))
        self._output = None

    def add_left_dependent(self, dependent, label=None):
        if not self._add_in_order:
            self.add_right_dependent(dependent, label)
            return
        self._dependents.insert(0, dependent)
        if self.use_labels:
            self._labels.insert(0, label)
            inputs = [dy.concatenate([self._head, l, d.get_state()]) for l, d in zip(self._labels, self._dependents)]
        else:
            inputs = [dy.concatenate([self._head, d.get_state()]) for d in self._dependents]
        self._state = self._init_state
        self._state = self._state.add_inputs(inputs)[-1]
        self._output = None


