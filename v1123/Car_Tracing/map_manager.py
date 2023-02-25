import matplotlib.pyplot as plt
import numpy as np

def file2map(path):
    file = open(path, 'r')
    map  = file.read().split('\n')

    police, criminals, landmarks, escape_pos = [], [], [], []

    for i in range(len(map)):
        _i = float(i)

        map[i] = list(map[i])
        for j in range(len(map[i])):
            _j = float(j)

            if map[i][j] == 'p':
                police.append((_i, _j))
            if map[i][j] == 'c':
                criminals.append((_i, _j))
            if map[i][j] == 'o':
                landmarks.append((_i, _j))
            if map[i][j] == 'e':
                escape_pos.append((_i, _j))

    return map, police, criminals, landmarks, escape_pos

def pic2file(path):
    pic = plt.imread(path)
    pic = np.delete(pic, 0, axis=2)
    pic = np.average(pic,axis=2)
    pic[pic<.15] = 255
    pic[pic!=255] = 0
    p = np.ndarray((200,250))
    for i in range(200):
        for j in range(250):
            v = 0
            for di in range(3):
                for dj in range(3):
                    v += pic[i*3+di][j*3+dj]
            if v > .2 * 9 * 255:
                p[i][j] = 255
            else:
                p[i][j] = 0
    f = open('map/city.txt','w')
    for l in p:
        for c in l:
            if c == 0:
                f.write('o')
            else:
                f.write('-')
        f.write('\n')
    f.close()
