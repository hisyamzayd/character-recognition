# adaline network, value based in ppt

# import random
import random
import numpy as np

# get data
def get_data(*filename) :
    data = []
    for i in filename :
        file = open(i, 'r').read()
        each = []
        for j in file :
            if j == '.' :
                each.append(-1)
            elif j == '#' :
                each.append(1)
        data.append(each)
    return data

# training process, to gain new weight and bias (data model)
def training_process(a, th, w, b, x_train, t_train) :
    # epoch
    epoch = 0
    # while iteration for each epoch
    while (True) :
        # increment epoch
        epoch += 1
        # bucket for substracted weight value within 1 epoch
        w_selisih = []
        # do loop for each row
        for i in range(len(t_train)) :
            # count y = v = b + sumof(XiWi)
            y = 0
            for j in range(len(w)) :
                y += np.float128(x_train[i][j]*w[j])
            y += b
            # save old weight value
            w_lama = w[:]
            # update new weight for each row
            for j in range(len(w)) :
                w[j] += np.float128(a*(t_train[i]-y)*x_train[i][j])
            b += a*(t_train[i]-y)
            # substract new weight value with old ones
            for j in range(len(w)) :
                w_selisih.append(w[j]-w_lama[j])
        # find max value from array of substracted weight
        w_selisih_max = max(w_selisih)
        # check if most substracted value less than threshold, then stop the while iteration
        if w_selisih_max < th or epoch > 1000 :
            # return last change of weight and bias
            return w, b

# testing process
def testing_process(new_w, new_b, x_test, t_test) :
    # boolean array of result
    result_bool = []
    # array result of target
    result_target = []
    # do loop for each row x
    for i in range(len(t_test)) :
        # count v = b + sumof(XiWi)
        v = 0
        for j in range(len(new_w)) :
            v += x_test[i][j]*new_w[j]
        v += new_b
        # do 'activation' function and add value to result_target
        if v >= 0 :
            y = 1
        else :
            y = -1
        result_target.append(y)
        # check each row target with y
        if y == t_test[i] :
            result_bool.append(True)
        else :
            result_bool.append(False)
    # return expected target, result target, and result bool
    return t_test, result_target, result_bool

# init variable
alpha = 0.1
threshold = 0.1
x1x2_for_train = [[1,1], [1,-1], [-1,1], [-1,-1]] #logika AND
target_for_train = [1, -1, -1, -1]
weight = [0.1, 0.2]
# x1x2_for_train = get_data('data_train/o1.txt', 'data_train/o2.txt', 'data_train/o3.txt', 'data_train/o4.txt', 'data_train/x1.txt', 'data_train/x2.txt', 'data_train/x3.txt', 'data_train/x4.txt')
# target_for_train = [1, 1, 1, 1, -1, -1, -1, -1]
# weight = [round(random.uniform(0.0, 1.0), 1) for i in x1x2_for_train[0]]
bias = 0.5
print('old weight')
print(weight)

# do training process, gain new value of weight and bias (data model)
new_weight, new_bias = training_process(alpha, threshold, weight, bias, x1x2_for_train, target_for_train)
print('new weight, and bias')
print(new_weight, new_bias, '\n')

# x and target value for testing
x1x2_for_test = [[1,-1], [1,1], [-1,-1], [-1,1]] #logika AND
target_for_test = [-1, 1, -1, -1]
# x1x2_for_test = get_data('data_test/o.txt', 'data_test/x.txt')
# target_for_test = [1, -1]

print('expected target, target result, and the boolean')
print(testing_process(new_weight, new_bias, x1x2_for_test, target_for_test))

