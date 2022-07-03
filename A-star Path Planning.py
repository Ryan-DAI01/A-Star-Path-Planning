import matplotlib.pyplot as plt
from heapq import *
import numpy as np
import time
import cv2

dir = [(0, 1), (1, 1), (1, 0), (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)]


def heuristic(a, b):  # a, b: 点  Form: 欧式距离
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    # return abs(a[0] - b[0]) + abs(a[1] - b[1])


def Path_Planning(map, start, end):
    open_list = []
    close_list = []
    Father_map = {}
    G_map = {start: 0}
    F_map = {start: heuristic(start, end)}  # F = G + H

    heappush(open_list, (F_map[start], start))

    while open_list:
        node = heappop(open_list)[1]  # open_list中最优点
        if node == end:  # node即为终点，返回路径
            path = [node]
            while node in Father_map:
                node = Father_map[node]
                path.append(node)
            return path
        for i, j in dir:
            new_node = (node[0] + i, node[1] + j)
            if new_node[0]<0 or new_node[0]>=map.shape[0] or new_node[1]<0 or new_node[1]>=map.shape[1]:  #判断是否越界
                continue
            if map[new_node[0]][new_node[1]] == 1:  # 判断是否为障碍物
                continue
            if new_node in close_list:  # 判断是否在close_list中
                continue

            tmp_g = G_map[node] + heuristic(node, new_node)
            tmp_f = tmp_g + heuristic(new_node, end)
            if new_node not in [k[1] for k in open_list]:  # 新节点已经在open_list中
                G_map[new_node] = tmp_g
                F_map[new_node] = tmp_f
                heappush(open_list, (F_map[new_node], new_node))
                Father_map[new_node] = node
            elif tmp_g < G_map[new_node]:  # 如果新路径比原有路径好
                G_map[new_node] = tmp_g
                F_map[new_node] = tmp_f
                Father_map[new_node] = node
            else:
                continue
        close_list.append(node)


if __name__ == "__main__":
    begin_time = time.time()
    img = cv2.imread('map.png', cv2.IMREAD_COLOR)
    resized = cv2.resize(img, (100, 100))  # 缩放图片
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)  # 灰度化
    for i in range(gray.shape[0]):  # 二值化
        for j in range(gray.shape[1]):
            if gray[i][j] >= 127:
                gray[i][j] = 0
            else:
                gray[i][j] = 1
    path = Path_Planning(gray, (2, 2), (97, 97))  # 路径规划(起点和终点的选取)
    end_time = time.time()
    duration = end_time - begin_time
    # 绘图
    for i in range(len(path)-1):
        cv2.line(resized, (path[i][1], path[i][0]), (path[i+1][1], path[i+1][0]), (218, 112, 214))
    plt.imshow(resized)
    plt.imsave('res.png', resized)
    plt.axis('off')
    plt.show()
    print("规划用时：%f s" % duration)
