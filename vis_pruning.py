import numpy as np
import matplotlib.pyplot as plt

def ret_data():
    mnist_g_10p_10p_10c_10c_x = np.array([99.01960784313727, 79.41176470588235, 59.80392156862745, 40.19607843137255, 20.588235294117645, 1.9411764705882355, 1.5490196078431373, 1.1372549019607843, 0.7647058823529412, 0.37254901960784315])
    mnist_g_10p_10p_10c_10c_y = np.array([98.48, 98.56, 98.34, 98.28, 98.35, 11.35, 11.35, 11.35, 11.35, 11.35])

    legends = ['a-group-10-10-p-10-10-c']
    return mnist_g_10p_10p_10c_10c_x, mnist_g_10p_10p_10c_10c_y, legends



if __name__== '__main__':

    data = ret_data()
    
    for plot_idx  in np.arange(0,len(data)-1,2):
        plt.plot(data[plot_idx], data[plot_idx+1])

    plt.legend(data[-1])

    plt.xlim([101.0, 8.5])
    plt.xlabel('% of weights remaining from original model')    
    plt.ylabel('Recognition performance (%)')
    plt.title('Comparison of performance variation when AlexNet is one-shot pruned, w/o retraining')

    plt.show()
