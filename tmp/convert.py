import os

with open('./nbsjfb.txt', encoding='UTF-8') as fp:
    for line in fp.readlines():
        if line != '\n':
            _line = line.strip('\n')
            _line = _line.split()
            print('{"name": "%s", "value": %d},' % (_line[0], int(_line[1])))