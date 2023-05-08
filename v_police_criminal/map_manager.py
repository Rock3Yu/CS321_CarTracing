import matplotlib.pyplot as plt
import numpy as np

def file2map(path):
    file = open(path, 'r')
    map_data = list(filter(None, file.read().split('\n')))

    size = np.fromstring(map_data[0], sep=',', dtype=int)
    police = np.fromstring(map_data[1], sep=',', dtype=int)
    criminals = np.fromstring(map_data[2], sep=',', dtype=int)
    landmarks = []
    for line in map_data[3:-2]:
        line = list(filter(None, line.split(';')))
        arr = np.empty((len(line), 2))
        for i in range(len(line)):
            arr[i] = np.fromstring(line[i], sep=',')
        landmarks.append(arr)
    spp = list(filter(None, map_data[-2].split(';')))
    spawn_points = np.empty((len(spp), 2))
    for i in range(len(spp)): spawn_points[i] = np.fromstring(spp[i], sep=',')
    escape_pos = int(map_data[-1])

    return size, police, criminals, landmarks, spawn_points, escape_pos

'''
size_x,size_y
radar_p0,radar_p1,...,radar_pn,
radar_c0,radar_c1,...,radar_cm,
block0_p0_x,block0_p0_y;block0_p1_x,block0_p1_y;...;block0_pk0_x,block0_pk0_y;
block1_p0_x,block1_p0_y;block1_p1_x,block1_p1_y;...;block1_pk1_x,block1_pk1_y;
...
blockt_p0_x,blockt_p0_y;blockt_p1_x,blockt_p1_y;...;blockt_pkl_x,blockt_pkl_y;
spawn_p0_x,spawn_p0_y;spawn_p1_x,spawn_p1_y;...;spawn_pt_x,spawn_pt_y;
num_escape_pos
'''

def map_encode(map):
    """
    #space:0
    #landmark:1
    #police:2
    #criminal3
    #escape pos4
    map:(3,57,57)
    """
    map=np.transpose(map,(1,2,0))
    new_map=np.zeros((map.shape[0],map.shape[1],5))
    # print(np.all(map==np.array([255,255,255]),-1).shape)
    new_map[np.all(map==np.array([255,255,255]),-1),0]=1
    new_map[np.all(map==np.array([0,0,0]),-1),1]=1
    new_map[np.all(map==np.array([51,51,255]),-1),2]=1 
    new_map[np.all(map==np.array([255, 51, 51]),-1),3]=1 
    new_map[np.all(map==np.array([51, 204, 51]),-1),4]=1 
    return new_map.transpose((2,0,1))
    