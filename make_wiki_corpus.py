#!/usr/bin/env python3
"""
Creates a corpus from Wikipedia dump file.
Inspired by:
https://github.com/panyang/Wikipedia_Word2vec/blob/master/v1/process_wiki.py
"""

import sys
from itertools import chain
from gensim.corpora import WikiCorpus

def get_n_chars(wiki, n):
    nbchar = 0
    charbuffer = []
    for text in wiki.get_texts():
        for word in text:
            for char in chain(word, ' '):
                if nbchar < n:
                    nbchar += 1
                    charbuffer.append(char)
                else:
                    return charbuffer
    return charbuffer


def make_corpus(in_f, out_f, maxchar):
    wiki = WikiCorpus(in_f)
    charbuffer = get_n_chars(wiki, maxchar)
    with open(out_f, 'w') as output:
        output.write(''.join(charbuffer))


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('Usage: python make_wiki_corpus.py <wikipedia_dump_file> <processed_text_file> <number_of_characters>')
        sys.exit(1)
    in_f = sys.argv[1]
    out_f = sys.argv[2]
    maxchar = int(sys.argv[3])
    make_corpus(in_f, out_f, maxchar)
