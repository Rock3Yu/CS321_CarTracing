import ast
import numpy as np

EPOCH = 300


def curve_fitting(data, degree):
    import matplotlib.pyplot as plt
    import numpy as np

    for i in range(4):
        data_x = np.linspace(1, EPOCH, EPOCH)
        data_y = data[i]

        # 使用matplotlib绘制折线图
        data_x = np.array(data_x).astype(int)
        # plt.scatter(data_x, data_y, marker='o')  # 添加数据点
        # 绘制曲线
        poly = np.polyfit(data_x, data_y, deg=degree)
        y_value = np.polyval(poly, data_x)
        plt.plot(data_x, y_value)

    plt.legend()
    plt.show()


def line_chart(data):
    import matplotlib.pyplot as plt

    # 注意注意！！！本地跑！！！否则渲染看不到
    # 从 1 到 40 平均分成40份
    x = np.linspace(1, EPOCH, EPOCH)  # linspace 是numpy里的一个函数 用于生成一组等区间的数值  # 40 是轮数，训练多少轮就是多少
    y0 = data[0]
    plt.plot(x, y0)

    y1 = data[1]
    plt.plot(x, y1)

    y2 = data[2]
    plt.plot(x, y2)

    y3 = data[3]
    plt.plot(x, y3)

    plt.show()

    # 外观调整
    # 颜色     color='red'
    # 点形状   marker='0'
    # 线性     linestyle='-'  # 虚线  ’--’
    # 多条直线的话多plt生成多条

def cul_reward(reward):
    N = 4 #agent number
    str1 = open('reward.txt', 'r').read()  # 文件名变了记得改！
    reward = ast.literal_eval(str1)
    reward2 = np.zeros((N, EPOCH))  # 固定长度请用np.array

    for ind, re in enumerate(reward):  #ind: epoch
        len1=len(re) #num of cycles
            
        
        for j in range(4):
            reward2[j][ind]=re[len1-1]
            reward2[j][ind] = reward2[j][ind] / len1
    print(reward2)
    reward = np.array(reward2)

    print(reward2.shape)
    
def main():
    N = 4 #agent number
    str1 = open('reward.txt', 'r').read()  # 文件名变了记得改！
    reward = ast.literal_eval(str1)
    reward2 = np.zeros((N, EPOCH))  # 固定长度请用np.array

    for ind, re in enumerate(reward):
        len1 = 0
        for i, r in enumerate(re):
            len1 += 1
            for j, v in enumerate(r):
                reward2[j][ind] += v
        for j in range(4):
            reward2[j][ind] = reward2[j][ind] / len1
    print(reward2)
    reward = np.array(reward2)

    print(reward2.shape)
    # print(reward)

    # line_chart(reward2)  # 直线图

    curve_fitting(reward2, 5)  # 拟合的曲线图，维度


if __name__ == '__main__':
    main()
