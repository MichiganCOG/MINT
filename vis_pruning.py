import numpy as np
import matplotlib.pyplot as plt

if __name__== '__main__':


    ## STL-10
    #oneshot_x   = np.array([100.0,     90.277778, 79.861111, 70.138889, 59.722222, 50.0,      40.277778, 29.861111, 20.138889, 9.722222])
    #oneshot_y   = np.array([42.100000, 39.925000, 38.687500, 37.162500, 32.200000, 30.987500, 38.100000, 36.137500, 31.575000, 20.512500])

    #a_x         = np.array([])
    #a_y         = np.array([])

    #b_x         = np.array([99.305556, 90.277778, 79.166667, 70.138889, 59.722222, 49.305556, 40.277778, 29.861111, 20.138889, 9.722222])
    #b_y         = np.array([42.062500, 42.225000, 41.612500, 30.425000, 29.225000, 27.187500, 29.262500, 23.912500, 25.037500, 10.0])

    #a_group_2_x = np.array([100.0,     88.888889, 78.472222, 65.972222, 64.583333, 45.138889, 37.500000, 25.000000, 22.222222, 17.361111])
    #a_group_2_y = np.array([42.100000, 42.100000, 36.775000, 38.550000, 41.837500, 19.162500, 29.925000, 18.612500, 13.212500, 10.0 ])

    #a_group_3_x = np.array([100.0,     89.583333, 80.555556, 69.444444, 57.638889, 47.222222, 39.583333, 38.888889, 20.833333, 6.944444])
    #a_group_3_y = np.array([42.100000, 42.100000, 36.862500, 38.350000, 43.800000, 43.562500, 38.075000, 27.950000, 17.125000, 10.000000])

    # CIFAR-10
    oneshot_x   = np.array([100.0,     90.277778, 79.861111, 70.138889, 59.722222, 50.0,      40.277778, 29.861111, 20.138889, 9.722222])
    oneshot_y   = np.array([77.190000, 77.610000, 74.830000, 68.800000, 57.860000, 56.750000, 56.630000, 55.960000, 22.180000, 9.870000])

    a_x         = np.array([])
    a_y         = np.array([])

    b_x         = np.array([])
    b_y         = np.array([])

    a_group_2_x = np.array([])
    a_group_2_y = np.array([])

    a_group_3_x = np.array([])
    a_group_3_y = np.array([])

    plt.plot(oneshot_x, oneshot_y)
    #plt.plot(b_x, b_y)
    #plt.plot(a_group_2_x, a_group_2_y)
    #plt.plot(a_group_3_x, a_group_3_y)
    plt.xlim([101.0, 8.5])
    plt.xlabel('Ratio of weights pruned')    
    plt.ylabel('Performance of AlexNet')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')

    plt.legend(['weight-based'])#,'b','a-group-2','a-group-3'])
    plt.show()
