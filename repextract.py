#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from itertools import chain

root = ET.parse('1999-05-17.xml').getroot()
articles = root.findall('./tei:text/tei:body/tei:div/tei:div/',
                        {'tei': 'http://www.tei-c.org/ns/1.0'})

alphabet = ' aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz'
# print(list(root))
# print(articles)

def filterspaces(iterable):
    prevwasspace = True
    for char in iterable:
        if char == ' ':
            if not prevwasspace:
                prevwasspace = True
                yield char
        else:
            yield char
            prevwasspace = False


charbuffer = (char
              for article in articles
              for paragraph in article.itertext()
              for char in paragraph.lower()
              if char in alphabet)

with open('1999-05-17.txt', 'w') as output:
    output.write(''.join(filterspaces(charbuffer)))
