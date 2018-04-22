import nltk
nltk.download('brown')
nltk.download('nonbreaking_prefixes')
nltk.download('perluniprops')
from nltk.corpus import brown
from nltk.tokenize.moses import MosesDetokenizer

mdetok = MosesDetokenizer()

def remove_brown_annotations(sentence):
    return mdetok.detokenize(
        ' '.join(sent).replace('``', '"')\
        .replace("''", '"')\
        .replace('`', "'").split(),
        return_str=True)


maxnbchar = 50000
currentnbchar = 0
charbuffer = []

alphabet = 'abcdefghijklmnopqrstuvwxyz '

for sent in brown.sents():
    for char in remove_brown_annotations(sent):
        if currentnbchar < maxnbchar and char in alphabet:
            charbuffer.append(char)
            currentnbchar += 1

output = 'brown50000.txt'
with open(output, "w") as text_file:
    text_file.write(''.join(charbuffer))
