import numpy as np
class UnionFind:
    def __init__(self, n):
        """长度为n的并查集"""
        self.uf = [-1] * (n + 1)  # 列表0位置空出
        self.sets_count = n  # 判断并查集里共有几个集合, 初始化默认互相独立

    def find(self, p):
        """查找p的根结点(祖先)"""
        r = p  # 初始p
        while self.uf[p] > 0:
            p = self.uf[p]
        while r != p:  # 路径压缩, 把搜索下来的结点祖先全指向根结点
            self.uf[r], r = p, self.uf[r]
        return p

    def union(self, p, q):
        """连通p,q 让q指向p"""
        proot = self.find(p)
        qroot = self.find(q)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:  # 负数比较, 左边规模更小
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]
            self.uf[qroot] = proot
        self.sets_count -= 1                   # 连通后集合总数减一

    def is_connected(self, p, q):
        """判断pq是否已经连通"""
        return self.find(p) == self.find(q)  # 即判断两个结点是否是属于同一个祖先

#im_binary() 方法把一个图像中等于target_value的值置为1，其余的置为0，实现二值化。

def im_binary(data: np.ndarray, target_value: int):
    return np.where(data == target_value, 1, 0)

#im_padding() 把图片周围补一圈0，这样才能用下面的算法。
def im_padding(data: np.ndarray):
    return np.pad(data, ((1, 1), (1, 1)), 'constant')


def first_pass(data, uf_set):
    offsets = [[-1, -1], [0, -1], [-1, 1], [-1,  0]]
    label_counter = 1
    for y in range(1, data.shape[0]-1):
        for x in range(1, data.shape[1]-1):
            if data[y, x] == 0:
                continue
            neighbor = []
            for offset in offsets:
                if data[y + offset[0], x + offset[1]] != 0:
                    neighbor.append(data[y + offset[0], x + offset[1]])
            neighbor = np.unique(neighbor)
            if len(neighbor) == 0:
                data[y, x] = label_counter
                label_counter += 1
            elif len(neighbor) == 1:
                data[y, x] = neighbor[0]
            else:
                # 邻居内有多重label, 这种情况要把最小值赋给data[y, x], 同时建立值之间的联系.
                data[y, x] = neighbor[0]
                for n in neighbor:
                    uf_set.union(int(neighbor[0]), int(n))


def second_pass(data, uf_set):
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if data[y, x] != 0:
                data[y, x] = uf_set.find(int(data[y, x]))

def count_patch(data, get_img=False):
    # 统计某一种类别的图斑数. 返回各个图斑的像素数，和结果图.
    ufSet = UnionFind(1000000)
    first_pass(data, ufSet)
    second_pass(data, ufSet)

    count_dic = {}
    for y in range(1, data.shape[0] - 1):
        for x in range(1, data.shape[1] - 1):
            if data[y, x] in count_dic:
                count_dic[data[y, x]] += 1
            else:
                count_dic[data[y, x]] = 1

    count_dic.pop(0)
    print(count_dic)
    flag=max(count_dic, key=count_dic.get)#求dict的最大索引对应的键值
    print(flag)
    data=im_binary(data,flag)
    if get_img:
        return  data
    else:
        return list(count_dic.values())