import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image

import csv
from scipy.interpolate import interp1d

file_configs = {
    'original_data_path': r'.\mit-bih-arrhythmia-database-1.0.0',
    'kaggle_data_path': r'.\mitbih_database',
    'hym_rym':r'.\rhythm_data'
}

def read_data(num):
    data = np.loadtxt(os.path.join(file_configs['kaggle_data_path'], str(num)+'.csv'), delimiter=',', skiprows=1)  # 跳过第一行表头
    channel1 = data[:, 1]  # 提取第二列 MLII
    channel1 -= 1024
    channel1 = (channel1 / 200).astype(np.float32)
    return channel1

def slice_data(num,beat_per_slice=20,offset=20):
    data = read_data(num)
    beat_sp = [0]
    lable = [0]
    with open(os.path.join(file_configs['kaggle_data_path'], str(num)+'annotations.txt')) as f:
        lines = f.readlines()
        for index in range(1,len(lines)):
            line = lines[index]
            line = line.split()
            beat_sp.append(int(line[1]))
            lable.append(line[2])
        res = []
        for index in range(1,len(beat_sp)-beat_per_slice,beat_per_slice):
            res.append(
                {
                    'data':data[beat_sp[index-1]+offset:beat_sp[index+beat_per_slice]-offset],
                    'label':lable[index:index+beat_per_slice]
                }
            )
        return res

def resample(sliced_data, spnum=2560):
    resampled_data = []
    for item in sliced_data:
        original_length = len(item['data'])
        # 创建原始数据的插值函数
        f = interp1d(np.linspace(0, 1, original_length), item['data'], kind='linear')
        # 生成新的等间距点并插值得到固定长度的数据
        resampled = f(np.linspace(0, 1, spnum))
        resampled_data.append({
            'data': resampled.astype(np.float32),
            'label': item['label']
        })
    return resampled_data
def make_data(num, output_file):
    sliced = slice_data(num)
    resampled_data = resample(sliced)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for item in resampled_data:
            writer.writerow(item['data'].tolist())
def make_data_for_hym(num,output_file):
    data = read_data(num)
    sliced_data = []
    with open(file_configs['hym_rym']+'\\'+str(num)+'rhythm.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            start = int(line[11])
            end = int(line[20])
            temp = data[start:end]
            original_length = len(temp)

            # 创建插值函数
            f = interp1d(np.linspace(0, 1, original_length), temp, kind='linear')

            # 生成固定长度（2560）的重采样数据
            resampled_temp = f(np.linspace(0, 1, 2560)).astype(np.float32)
            sliced_data.append(resampled_temp)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for item in sliced_data:
            writer.writerow(item.tolist())
def make_plot(num, output_dir):
    sliced = slice_data(num,beat_per_slice=1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, item in enumerate(sliced[1:]):
        ecg_data = item['data']

        # 创建绘图（128x128像素）
        plt.figure(figsize=(1.28, 1.28), dpi=100)
        plt.plot(ecg_data, color='black',linewidth = 0.4)  # 黑白线条
        plt.axis('off')  # 关闭坐标轴
        plt.tight_layout(pad=0)  # 紧密排版

        # 调整图像边缘留白为0
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
        # 提取原始 label 列表并进行字符替换
        new_labels = []
        for label in item['label']:
            if label == '|':
                new_labels.append('i')
            elif label == '/':
                new_labels.append('p')
            elif label == '"':
                new_labels.append('q')
            else:
                new_labels.append(label)

        # 拼接成字符串
        label_str = ''.join(new_labels)

        # 构建图像路径
        image_path = os.path.join(output_dir, f"{num}_{idx}_[{label_str}].png")

        # 保存图像并关闭
        plt.savefig(image_path, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 可选：转换为灰度图（虽然已经是黑白，但确保格式正确）
        img = Image.open(image_path).convert('L')
        img.save(image_path)

if __name__ == '__main__':


    dir = os.listdir(r'.\mitbih_database')
    dir = [f for f in dir if f.endswith('.csv')]
    for f in dir:
        print(f)
        num = int(f.split('.')[0])
        make_data_for_hym(num,r'.\csv'+'\\'+str(num)+'.csv')

   #     make_plot(num,r'.\plots')