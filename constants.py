root_form = '<root-form>'
root_pos = '<root-pos>'
unk_word = '<unk-word>'
none_word = '<none-word>'
none_pos = '<none-pos>'
none_dep = '<none-dep>'


class CONLL09:
    INDEX = 0
    FORM = 1
    LEMMA = 2
    PLEMMA = 3
    POS = 4
    PPOS = 5
    FEAT = 6
    PFEAT = 7
    HEAD = 8
    PHEAD = 9
    DEPREL = 10
    PDEPREL = 11
    root_token = ['0', root_form, '_', '_', root_pos, root_pos, '', '', '-1', '-1', none_dep, none_dep]


class CONLL06:
    INDEX = 0
    FORM = 1
    LEMMA = 2
    CPOS = 3
    POS = PPOS = 4  # To help keep data parsing general
    FEATS = 5
    HEAD = 6
    DEPREL = 7
    PHEAD = 8
    PDEPREL = 9
    root_token = ['0', root_form, '_', root_pos, root_pos, '', '-1', none_dep, '-1', none_dep]

class CONLLU:
    INDEX = 0
    FORM = 1
    LEMMA = 2
    UPOS = CPOS = 3
    XPOS = POS = PPOS = 4  # To help keep data parsing general
    FEATS = 5
    HEAD = PHEAD = 6
    DEPREL = PDEPREL = 7
    DEPS = 8
    MISC = 9
    root_token = ['0', root_form, '_', root_pos, root_pos, '', '-1', none_dep, '-1', none_dep]
