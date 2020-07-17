from sympy import *
import random
import numpy as np

xdata = np.array([[1.21080233, 1.91763022, 0.84189102],
                  [16.90783848, -2.18846433, -0.20549639],
                  [25.41711405, 7.98390437, 20.78739232],
                  [30.09521698, -14.64470851, 24.39972867],
                  [13.47591857, 43.33069247, 39.36503645],
                  [19.04462919, 46.83931677, 40.97510939],
                  [19.36014403, -21.76969205, -26.65076334],
                  [0.42409849, -32.53481834, 26.34011983],
                  [31.97285153, 85.34422982, -36.32665085],
                  [63.36590334, -9.02758881, -44.10226629]])
ydata = np.array([[20.44112772],
                  [79.95619596],
                  [210.75439539],
                  [161.24493424],
                  [325.93508504],
                  [369.87799685],
                  [-44.63309396],
                  [-43.69696385],
                  [378.58957275],
                  [173.23794093]])
w_hat = np.array([[4.9], [3.6], [2.5]])
b_hat = 5.5
# 数据导入完毕#
# print((x.dot(w_hat)+b_hat-y)<0.000001)##数据导入准确性测试

xmean = np.mean(xdata, axis=0)
xstd = np.std(xdata, axis=0)
xdata = (xdata - xmean) / xstd

iteration = 1000
lr = 30
the = [0, 0, 0, 0]# initialize

x1_, x2_, x3_, w1_, w2_, w3_, b_, y = symbols('x1_,x2_,x3_,w1_,w2_,w3_,b_,y')
w_ = [w1_, w2_, w3_]
x_ = [x1_, x2_, x3_]
z = (y - np.dot(w_, x_) - b_) ** 2
# print(z)
dthe = [diff(z, w1_), diff(z, w2_), diff(z, w3_), diff(z, b_)]
print(dthe)

lr_the = [0., 0., 0., 0.]
for i in range(iteration):
    grad = np.array([0., 0., 0., 0.])
    print('process : {}/{}'.format(i, iteration))
    n = random.randint(0, len(xdata) - 1)
    for ii in range(4):
        grad[ii] = dthe[ii].subs(
            [(x1_, xdata[n][0]), (x2_, xdata[n][1]), (x3_, xdata[n][2]), (y, ydata[n]), (w1_, the[0]), (w2_, the[1]), (w3_, the[2]),
             (b_, the[3])])
    lr_the += grad[0:4] ** 2
    for ii in range(4):
        the[ii] -= lr / ((lr_the[ii]) ** 0.5) * grad[ii]

w1 = the[0] / xstd[0]
w2 = the[1] / xstd[1]
w3 = the[2] / xstd[2]
b = the[3] - (w1 * xmean[0] + w2 * xmean[1] + w3 * xmean[2])

print(w1, w2, w3, b)
