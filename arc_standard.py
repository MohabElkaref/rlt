# coding=utf-8
from constants import *


class ArcStandard:
    SHIFT = 'SH'
    LEFT_ARC = 'LA'
    RIGHT_ARC = 'RA'

    SHIFT_IDX = 0
    LEFT_ARC_IDX = 1
    RIGHT_ARC_IDX = 2

    INVALID = -1
    EMPTY_LABEL = '<EMPTY-LABEL>'
    EMPTY_BUFFER = -1

    def __init__(self, sentence):
        self._sentence = sentence
        self._format = sentence.format
        self._buffer_index = 1
        self._stack = [0]
        self.transitions = []  # dep_id, head_id, transition, label

    def apply(self, transition, label):
        if transition == ArcStandard.RIGHT_ARC:  # (σ|i|j, β, A) => (σ|i, β, A∪{(i, j, k)})
            dep_id = self._stack.pop()
            head_id = self._stack[-1]
            self.transitions.append((dep_id, head_id, transition, label))
            self._sentence.add_arc(dep_id, head_id, label)
        elif transition == ArcStandard.LEFT_ARC:  # (σ|i|j, β, A) => (σ|j, β, A∪{(i, j, k)})
            head_id = self._stack[-1]
            dep_id = self._stack.pop(-2)
            self.transitions.append((dep_id, head_id, transition, label))
            self._sentence.add_arc(dep_id, head_id, label)
        elif transition == ArcStandard.SHIFT:  # (σ, i|β, A) => (σ|i, β, A∪{(i, j, k)})
            self.transitions.append((ArcStandard.INVALID, self._buffer_index, transition, ArcStandard.EMPTY_LABEL))
            self._stack.append(self._buffer_index)
            self._buffer_index += 1
            if self._buffer_index >= len(self._sentence):
                self._buffer_index = ArcStandard.EMPTY_BUFFER
        return self.is_done()

    def apply_gold(self):
        s0 = self._stack[-1]
        if len(self._stack) >= 2:
            s1 = self._stack[-2]
        else:
            s1 = -1
        if s1 > 0 and self._sentence[s1][self._format.HEAD] == s0:
            return self.apply(ArcStandard.LEFT_ARC, self._sentence[s1][self._format.DEPREL])
        if 0 <= s1 == self._sentence[s0][self._format.HEAD]:
            possible = True
            if self._buffer_index != ArcStandard.EMPTY_BUFFER:
                for i in range(self._buffer_index, len(self._sentence)):
                    if self._sentence[i][self._format.HEAD] == s0:
                        possible = False
                        break
            if possible:
                return self.apply(ArcStandard.RIGHT_ARC, self._sentence[s0][self._format.DEPREL])
        return self.apply(ArcStandard.SHIFT, ArcStandard.EMPTY_LABEL)

    def is_done(self):
        if len(self._stack) == 1 and self._buffer_index == ArcStandard.EMPTY_BUFFER:
            return True
        return False

    def get_legal_transitions(self):
        transitions = [False, False, False]
        if len(self._stack) > 2:
            transitions[ArcStandard.LEFT_ARC_IDX] = True
            transitions[ArcStandard.RIGHT_ARC_IDX] = True
        elif len(self._stack) == 2 and self._buffer_index == ArcStandard.EMPTY_BUFFER:
            transitions[ArcStandard.RIGHT_ARC_IDX] = True
        if self._buffer_index != ArcStandard.EMPTY_BUFFER:
            transitions[ArcStandard.SHIFT_IDX] = True
        return transitions

    def get_best_transition(self, scores, labels):
        legal_transitions = self.get_legal_transitions()
        max_idx = -1
        transition = ['', '']
        if legal_transitions[ArcStandard.SHIFT_IDX]:
            max_idx = 0
            transition = [ArcStandard.SHIFT, ArcStandard.EMPTY_LABEL]
        if legal_transitions[ArcStandard.LEFT_ARC_IDX]:
            if labels:
                for i in range(1, len(labels) + 1):
                    if max_idx == -1 or scores[i] >= scores[max_idx]:
                        max_idx = i
                        transition = [ArcStandard.LEFT_ARC, labels[i - 1]]
            else:
                if max_idx == -1 or scores[1] >= scores[max_idx]:
                    max_idx = 1
                    transition = [ArcStandard.LEFT_ARC, ArcStandard.EMPTY_LABEL]
        if legal_transitions[ArcStandard.RIGHT_ARC_IDX]:
            if labels:
                for i in range(len(labels) + 1, 2 * len(labels) + 1):
                    if max_idx == -1 or scores[i] >= scores[max_idx]:
                        max_idx = i
                        transition = [ArcStandard.RIGHT_ARC, labels[i - len(labels) - 1]]
            else:
                if max_idx == -1 or scores[2] >= scores[max_idx]:
                    max_idx = 2
                    transition = [ArcStandard.RIGHT_ARC, ArcStandard.EMPTY_LABEL]
        return transition, max_idx

    def get_best_transition2(self, transition_scores, label_scores, labels):
        legal_transitions = self.get_legal_transitions()
        max_idx = -1
        transition = ['', '']
        best_label = label_scores.index(max(label_scores))
        if legal_transitions[ArcStandard.SHIFT_IDX]:
            max_idx = 0
            transition = [ArcStandard.SHIFT, ArcStandard.EMPTY_LABEL]
        if legal_transitions[ArcStandard.LEFT_ARC_IDX]:
            if max_idx == -1 or transition_scores[1] >= transition_scores[max_idx]:
                max_idx = 1
                transition = [ArcStandard.LEFT_ARC, labels[best_label]]
        if legal_transitions[ArcStandard.RIGHT_ARC_IDX]:
            if max_idx == -1 or transition_scores[2] >= transition_scores[max_idx]:
                max_idx = 2
                transition = [ArcStandard.RIGHT_ARC, labels[best_label]]
        return transition, max_idx

    def get_best_transition3(self, transition_scores, left_label_scores, right_label_scores, labels):
        legal_transitions = self.get_legal_transitions()
        max_idx = -1
        transition = ['', '']
        best_left_label = left_label_scores.index(max(left_label_scores))
        best_right_label = right_label_scores.index(max(right_label_scores))
        if legal_transitions[ArcStandard.SHIFT_IDX]:
            max_idx = 0
            transition = [ArcStandard.SHIFT, ArcStandard.EMPTY_LABEL]
        if legal_transitions[ArcStandard.LEFT_ARC_IDX]:
            if max_idx == -1 or transition_scores[1] >= transition_scores[max_idx]:
                max_idx = 1
                transition = [ArcStandard.LEFT_ARC, labels[best_left_label]]
        if legal_transitions[ArcStandard.RIGHT_ARC_IDX]:
            if max_idx == -1 or transition_scores[2] >= transition_scores[max_idx]:
                max_idx = 2
                transition = [ArcStandard.RIGHT_ARC, labels[best_right_label]]
        return transition, max_idx

    def get_best_extended_transition(self, scores, labels, tags, instance_tags, words, instance_words):
        legal_transitions = self.get_legal_transitions()
        max_idx = -1
        chosen_pos_idx = -1
        head_pos_idx = len(tags) - 1
        chosen_lex_idx = -1
        head_lex_idx = words[none_word]
        transition = ['', '']
        s0i = tags.index(instance_tags[0])
        s1i = tags.index(instance_tags[1])
        b0i = tags.index(instance_tags[2])
        s0w = words[instance_words[0]] if instance_words[0] in words.keys() else words[unk_word]
        s1w = words[instance_words[1]] if instance_words[1] in words.keys() else words[unk_word]
        b0w = words[instance_words[2]] if instance_words[2] in words.keys() else words[unk_word]
        if legal_transitions[ArcStandard.SHIFT_IDX]:
            max_idx = b0i
            chosen_pos_idx = b0i
            chosen_lex_idx = b0w
            transition = [ArcStandard.SHIFT, ArcStandard.EMPTY_LABEL]
        if legal_transitions[ArcStandard.LEFT_ARC_IDX]:
            for i in range(1, len(labels) + 1):
                idx = i * (len(tags)-1) + s1i
                if max_idx == -1 or scores[idx] >= scores[max_idx]:
                    max_idx = idx
                    chosen_pos_idx = s1i
                    head_pos_idx = s0i
                    chosen_lex_idx = s1w
                    head_lex_idx = s0w
                    transition = [ArcStandard.LEFT_ARC, labels[i-1]]
        if legal_transitions[ArcStandard.RIGHT_ARC_IDX]:
            for i in range(len(labels) + 1, 2 * len(labels) + 1):
                idx = i * (len(tags)-1) + s0i
                if idx == len(scores):
                    print 's0i is ' + str(s0i) + ' and the instance is ' + instance_tags[0]
                if max_idx == -1 or scores[idx] >= scores[max_idx]:
                    max_idx = idx
                    chosen_pos_idx = s0i
                    head_pos_idx = s1i
                    chosen_lex_idx = s0w
                    head_lex_idx = s1w
                    transition = [ArcStandard.RIGHT_ARC, labels[i - len(labels) - 1]]
        return transition, max_idx, chosen_pos_idx, head_pos_idx, chosen_lex_idx, head_lex_idx

    def evaluate(self):
        return self._sentence.evaluate()

    def get_tree_string(self):
        return self._sentence.get_tree_string()

    def extract_instances(self, features):
        instances = [''] * len(features)
        for i in range(0, len(features)):
            index = 0
            char_pos = 1
            offset = int(features[i][char_pos])
            char_pos += 1

            if features[i][0] == 's':
                if len(self._stack) > offset:
                    index = self._stack[-1 - offset]
                else:
                    index = -1
            elif features[i][0] == 'b':
                if self._buffer_index + offset < 0 or self._buffer_index + offset >= len(self._sentence):
                    index = -1
                else:
                    index = self._buffer_index + offset

            if features[i][char_pos] == 'n':
                char_pos += 2
            elif features[i][char_pos] == 'p':
                char_pos += 1
                index += int(features[i][char_pos])
                char_pos += 1
                if index >= len(self._sentence):
                    index = -1
            elif features[i][char_pos] == 'm':
                char_pos += 1
                index -= int(features[i][char_pos])
                char_pos += 1
                if index < 0:
                    index = -1
            elif features[i][char_pos] == 'h':
                char_pos += 2
                if index >= 0:
                    index = self._sentence[index][self._format.PHEAD]
            elif features[i][char_pos] == 'c':
                char_pos += 1
                if features[i][char_pos] == '-':
                    char_pos += 1
                    index = self._sentence.get_leftmost_child(index, int(features[i][char_pos]))
                else:
                    index = self._sentence.get_rightmost_child(index, int(features[i][char_pos]))
                char_pos += 1
            elif features[i][char_pos] == 'g':
                char_pos += 1
                if features[i][char_pos] == '-':
                    char_pos += 1
                    index = self._sentence.get_leftmost_child(index, int(features[i][char_pos]))
                    index = self._sentence.get_leftmost_child(index, 1)
                else:
                    index = self._sentence.get_rightmost_child(index, int(features[i][char_pos]))
                    index = self._sentence.get_rightmost_child(index, 1)
                char_pos += 1

            if index < 0:
                if features[i][char_pos] == 'w':
                    instances[i] = none_word
                elif features[i][char_pos] == 't':
                    instances[i] = none_pos
                elif features[i][char_pos] == 'l':
                    instances[i] = none_dep
                else:
                    instances[i] = none_word
            else:
                if features[i][char_pos] == 'w':
                    instances[i] = self._sentence[index][self._format.FORM]
                elif features[i][char_pos] == 't':
                    instances[i] = self._sentence[index][self._format.PPOS]
                elif features[i][char_pos] == 'l':
                    instances[i] = self._sentence[index][self._format.PDEPREL]

        return instances

    def extract_indexes(self, features):
        indexes = [-1] * len(features)
        for i in range(0, len(features)):
            index = 0
            char_pos = 1
            offset = int(features[i][char_pos])
            char_pos += 1

            if features[i][0] == 's':
                if len(self._stack) > offset:
                    index = self._stack[-1 - offset]
                else:
                    index = -1
            elif features[i][0] == 'b':
                if self._buffer_index + offset < 0 or self._buffer_index + offset >= len(self._sentence):
                    index = -1
                else:
                    index = self._buffer_index + offset

            if features[i][char_pos] == 'n':
                char_pos += 2
            elif features[i][char_pos] == 'p':
                char_pos += 1
                index += int(features[i][char_pos])
                char_pos += 1
                if index >= len(self._sentence):
                    index = -1
            elif features[i][char_pos] == 'm':
                char_pos += 1
                index -= int(features[i][char_pos])
                char_pos += 1
                if index < 0:
                    index = -1
            elif features[i][char_pos] == 'h':
                char_pos += 2
                if index >= 0:
                    index = self._sentence[index][self._format.PHEAD]
            elif features[i][char_pos] == 'c':
                char_pos += 1
                if features[i][char_pos] == '-':
                    char_pos += 1
                    index = self._sentence.get_leftmost_child(index, int(features[i][char_pos]))
                else:
                    index = self._sentence.get_rightmost_child(index, int(features[i][char_pos]))
                char_pos += 1
            elif features[i][char_pos] == 'g':
                char_pos += 1
                if features[i][char_pos] == '-':
                    char_pos += 1
                    index = self._sentence.get_leftmost_child(index, int(features[i][char_pos]))
                    index = self._sentence.get_leftmost_child(index, 1)
                else:
                    index = self._sentence.get_rightmost_child(index, int(features[i][char_pos]))
                    index = self._sentence.get_rightmost_child(index, 1)
                char_pos += 1

            if index < 0:
                indexes[i] = -1
            else:
                indexes[i] = index

        return indexes

    @staticmethod
    def get_gold_parse(sentence):
        parse = ArcStandard(sentence)
        while not parse.is_done():
            parse.apply_gold()
        return parse

    def get_stack(self, get_pos=False):
        stack = []
        if get_pos:
            for i in self._stack:
                stack.append(self._sentence[i][self._format.PPOS])
        else:
            for i in self._stack:
                stack.append(self._sentence[i][self._format.FORM])
        return stack

    def get_buffer(self, get_pos=False):
        buff = []
        if self._buffer_index is not ArcStandard.EMPTY_BUFFER:
            if get_pos:
                for i in range(self._buffer_index, len(self._sentence)):
                    buff.append(self._sentence[i][self._format.PPOS])
            else:
                for i in range(self._buffer_index, len(self._sentence)):
                    buff.append(self._sentence[i][self._format.FORM])
        return buff
