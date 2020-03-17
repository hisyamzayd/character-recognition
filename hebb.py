# data train
data = [[1,1], [1,-1], [-1,1], [-1,-1]] #logika and
target = [1, -1, -1, -1]

# weight and bias
weight = [0, 0]
bias = 0

# hebb learning cukup 1 epoch
def learning_hebb(d, t, w, b) :
    # update weight sebanyak row data
    for i in range(len(data)) :
        for j in range(len(w)) :
            w[j] += (d[i][j]*t[i])
        b += t[i]
    return w[:], b

# testing hebb network
def testing_hebb(d_test, t_test, new_w, new_b) :
    result_value = []
    result_bool = []
    # v = sumof(XiWi) + b
    for i in range(len(d_test)) :
        v = 0
        for j in range(len(new_w)) :
            v += (d_test[i][j]*new_w[j])
        v += new_b
        # lakukan cek aktivasi
        if v >= 0 :
            y = 1
        else :
            y = -1
        result_value.append(y)
        # cek boolean
        if y == t_test[i] :
            result_bool.append(True)
        else :
            result_bool.append(False)
    return t_test[:], result_value[:], result_bool[:]

# do the training
new_w, new_b = learning_hebb(data, target, weight, bias)

# data testing
data_test = [[-1,-1], [1,-1], [-1,1], [1,1]] #logika and
target_test = [-1, -1, -1, 1]

# do the testing
expec_result, real_result, boolean = testing_hebb(data_test, target_test, new_w, new_b)

# print the result
print('Expt Result\tReal Result\tBool')
print('%s\t%s\t%s' % (expec_result, real_result, boolean))