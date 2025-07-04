
import numpy as np
import os
import matplotlib.pyplot as plt
from pprint import pprint
from PIL import Image
import re


# env_settings:
file_configs = {
    'mit_bih_path': r'.\mit-bih-arrhythmia-database-1.0.0',
    'labels_path': r'.\mitbih_database',
    'num':100,
}
file_configs['head_file'] = str(file_configs['num'])+'.hea'
file_configs['data_file'] = str(file_configs['num'])+'.dat'
file_configs['labels_file'] = str(file_configs['num'])+'annotations.txt'
configs = {
    'plot_path': r'.\plots'
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
    temp1 = 0
    temp2 = 0
    for i in range(0, len(data), 3):
        temp1 = ((data[i+1] & 0xf0)<<4 )+ data[i]
        temp2 = ((data[i+1] & 0x0f)<<8 ) + data[i+2]
        channel1.append(temp1)
        channel2.append(temp2)
    channel1 = np.array(channel1)
    channel1 = channel1.astype(np.float32)

    channel2 = np.array(channel2)
    channel2 = channel2.astype(np.float32)
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
        data = data.astype(np.int16)
        channel1,channel2 = conv_format212(data)
        channel1 -= data_configs['channel1']['zero_offset']
        channel1 /= data_configs['channel1']['gain']
        channel2 -= data_configs['channel2']['zero_offset']
        channel2 /= data_configs['channel2']['gain']
        return channel1,channel2


def draw_plot(channel1, channel2, data_configs, samples_per_page=1080):
    """
    将心电数据绘制成波形图，自动分页显示。
    参数:
        channel1 (list or np.ndarray): 第一个通道的数据
        channel2 (list or np.ndarray): 第二个通道的数据
        data_configs (dict): 配置信息，包含采样率等
        samples_per_page (int): 每页显示的采样点数，默认 720 点
    """
    sample_rate = data_configs['sample_rate']
    total_samples = len(channel1)
    pages = (total_samples + samples_per_page - 1) // samples_per_page

    time_per_page = samples_per_page / sample_rate
    time_axis = np.linspace(0, time_per_page, samples_per_page)

    for page in range(pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, total_samples)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        fig.suptitle(
            f'ECG Signal - Page {page + 1} (Time: {start_idx / sample_rate:.1f}s - {end_idx / sample_rate:.1f}s)')

        # Channel 1
        ax1.plot(time_axis[:end_idx - start_idx], channel1[start_idx:end_idx], label='Channel 1', color='blue')
        ax1.set_title('Channel 1')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (mV)')
        ax1.grid(True)

        # Channel 2
        ax2.plot(time_axis[:end_idx - start_idx], channel2[start_idx:end_idx], label='Channel 2', color='green')
        ax2.set_title('Channel 2')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (mV)')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
        plt.close(fig)  # ⬅️ 关键：关闭当前 figure，防止内存泄漏

def slice_data(channel,labels_file):
    #      Time   Sample   Type  Sub Chan  Num	Aux
    sliced_data = []
    with open(labels_file, 'r') as f:
        lines = f.readlines()
        labels = []
        for line in lines[1:]:
            label = {}
            temp = line.split()
            label['Time'] =  temp[0]
            label['Sample'] = temp[1]
            label['Type'] = temp[2]
            labels.append(label)
    for index in range(1,len(labels)-1):
        start_sample = int(labels[index-1]['Sample']) + 50
        end_sample = int(labels[index+1]['Sample']) - 50
        sliced_data.append(
            {'label': labels[index]['Type'], 'data':channel[start_sample:end_sample]})
    return sliced_data

def save_sliced_data_as_images(sliced_data,file_configs, output_dir=r'.\plots\labeled_data'):
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
        plt.figure(figsize=(1.28, 1.28), dpi=100)
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
        resized_image = np.array(Image.fromarray(gray_image).resize((128, 128)))

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
    for index in range(230,234+1):
        file_configs['num'] = index
        print(index)
        file_configs['head_file'] = str(file_configs['num']) + '.hea'
        file_configs['data_file'] = str(file_configs['num']) + '.dat'
        file_configs['labels_file'] = str(file_configs['num']) + 'annotations.txt'
        channel1,channel2 = read_data(file_configs)
        sliced_data = slice_data(channel1,os.path.join(file_configs['labels_path'],file_configs['labels_file']))
        save_sliced_data_as_images(sliced_data,file_configs)

