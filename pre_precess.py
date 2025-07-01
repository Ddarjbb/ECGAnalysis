import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image

file_configs = {
    'mit_bih_path': r'.\mit-bih-arrhythmia-database-1.0.0',
    'labels_path': r'.\mitbih_database',
    'num':209,
}
file_configs['head_file'] = str(file_configs['num'])+'.hea'
file_configs['data_file'] = str(file_configs['num'])+'.dat'
file_configs['labels_file'] = str(file_configs['num'])+'annotations.txt'
configs = {
    'plot_path': r'.\hymplots'
}
def read_data_configs(file_configs):
    head_file_path =  os.path.join(file_configs['mit_bih_path'], file_configs['head_file'])
    data_configs = {}
    with open(head_file_path, 'r') as f:
        lines = f.readlines()
        index = 1
        for line in lines:
            config = line.split()
            # print(index, config)
            if config[0] == '#':
                break
            if index == 1 :
                data_configs['file_name'] = config[0]
                data_configs['channel_count'] = int(config[1])
                data_configs['sample_rate'] = int(config[2])
                data_configs['sample_points_count'] = int(config[3])
            if index > 1:
                channel_info = {
                'formal': config[1],
                'gain': int(config[2]),
                'resolution': config[3],
                'zero_offset': int(config[4])
                }
                data_configs['channel' + str(index - 1)] = channel_info
            index += 1
        return data_configs
def conv_format212(data):
    channel1 = []
    channel2 = []

    for i in range(0, len(data) - 2, 3):  # 防止越界读取
        byte1 = int(data[i])
        byte2 = int(data[i + 1])
        byte3 = int(data[i + 2])

        temp1 = ((byte2 & 0xf0) << 4) | byte1
        temp2 = ((byte2 & 0x0f) << 8) | byte3

        channel1.append(temp1)
        channel2.append(temp2)

    channel1 = np.array(channel1, dtype=np.int32)
    channel2 = np.array(channel2, dtype=np.int32)
    return channel1, channel2

def read_data(file_configs, data_configs =  None):
    if data_configs is None:
        data_configs = read_data_configs(file_configs)
    pprint(data_configs)
    data_file_path = os.path.join(file_configs['mit_bih_path'], file_configs['data_file'])
    head_file_path =  os.path.join(file_configs['mit_bih_path'], file_configs['head_file'])
    data_configs = read_data_configs(file_configs)
    with open(data_file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data = data.astype(np.int32)
        channel1,channel2 = conv_format212(data)
        channel1 -= data_configs['channel1']['zero_offset']
        channel2 -= data_configs['channel2']['zero_offset']
        channel1 = (channel1 / data_configs['channel1']['gain']).astype(np.float32)
        channel2 = (channel2 / data_configs['channel2']['gain']).astype(np.float32)

        return channel1,channel2

def slice_data(slice_config_file,channel):
    sliced_data = []

    with open(slice_config_file,'r') as f:
        lines = f.readlines()
        sliced_data = []
        for line in lines:
            temp = {}
            line = line.split()
            temp['beat_type'] = line[1:20]
            temp['start_sample'] = int(line[21])
            temp['end_sample'] =  int(line[40])
            temp['data'] = channel[int(temp['start_sample']):int(temp['end_sample'])]
            temp['label'] = line[41]
            #print(temp)
            sliced_data.append(temp)
    return sliced_data
def save_sliced_data_as_images(sliced_data,file_configs, output_dir=r'.\hymplots\labeled_data'):
    """
    将 sliced_data 转换为 128x128 灰度图像并保存。

    参数:
        sliced_data (list): 包含多个 ECG 片段的列表，每个片段是一个一维 NumPy 数组。
        output_dir (str): 图像保存的路径。
    """
    # 创建目标文件夹
    output_dir = os.path.join(output_dir,str(file_configs['num']))
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(sliced_data):
        # 绘制波形图（无坐标轴和边距）
        plt.figure(figsize=(1.28*10, 1.28), dpi=100)
        plt.plot(segment['data'], color='black',linewidth = 0.5)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # 将图像转换为 NumPy 数组
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # H x W x 3

        # 关闭当前图形以释放内存
        plt.close()

        # 转换为灰度图
        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  # RGB to grayscale
        gray_image = (gray_image - gray_image.min()) / (gray_image.max() - gray_image.min()) * 255
        gray_image = gray_image.astype(np.uint8)

        # 缩放为 128x128
        resized_image = np.array(Image.fromarray(gray_image).resize((128*10, 128)))

        # 保存图像
        label = segment['label']
        if label == '|':
            label = 'i'
        if label == r'/':
            label = 'p'
        if label == '"':
            label = 'q'
        output_path = os.path.join(output_dir, f"{i}_{label}.png")
        Image.fromarray(resized_image).save(output_path)

    print(f"✅ {len(sliced_data)} 张图像已保存至：{output_path}")

if __name__ == '__main__':
    channel1,channel2 = read_data(file_configs)
    print(channel1)
    sliced_data = slice_data('208rhythm.txt',channel1)
    save_sliced_data_as_images(sliced_data,file_configs)