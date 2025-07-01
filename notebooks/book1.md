# ECG 心电信号处理

## 1. MIT-BIH数据库
由美国麻省理工学院（MIT）和贝斯以色列医院（Beth Israel Hospital）联合建立。这个数据库包含了48个半小时常驻ECG记录，来自47名受试者，记录了各种心律失常现象。这些数据已被广泛用于心律分析研究和心律失常检测器的评估。MIT-BIH数据库的采样频率为360Hz，分辨率为11位，每个记录持续时间超过30分钟，采用Format 212格式存储。此外，该数据库还提供了人工标注的心拍位置和类型信息，方便研究者使用。MIT-BIH数据库是免费提供的，可以通过PhysioNet网站下载。

## 2. MIT-BIH数据库组成
心律失常数据库每一个数据记录都包含三个文件，“.hea”、“.dat”和“.atr”

.hea 为头文件，记录文件名、导联数、采样率、数据点数

.dat 为数据文件，采用212格式进行存储。

.atr 为注释文件，记录了心电专家对相应的心电信号的诊断信息。


### .hea 文件
这个文件是描述文件，记录了数据文件的基本信息。

第一行为记录行： 

文件名（如：100） 通道数（如：2代表二通道） 采样点数目（如：3600000 代表3600000个采样点） 

后面两行为信号技术规范：

数据存储方式（如：Format212） 增益（如：200 代表：200ADC uints/mV） 分辨率（如：11表示11位） 零值（如1024）

### .dat 文件
这个文件是数据文件，采用212格式进行存储。

### .atr 文件
为注释文件，记录了心电专家对相应的心电信号的诊断信息。
由于该文件读取规则非常复杂，我们采用Kaggle网站上整理好的标注数据：[MIT-BIH数据库](https://www.kaggle.com/datasets/mondejar/mitbih-database?resource=download)

## 3. 数据的读取

### 3.1 获取头文件信息

```python
import os
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
```
根据头文件的格式读取数据文件信息，返回到一个字典中。

### 3.2 数据的读取
```python
import numpy as np
import os
import pprint
def read_data(file_configs, data_configs =  None):
    if data_configs is None:
        data_configs = read_data_configs(file_configs)
    data_file_path = os.path.join(file_configs['mit_bih_path'], file_configs['data_file'])
    head_file_path =  os.path.join(file_configs['mit_bih_path'], file_configs['head_file'])
    data_configs = read_data_configs(file_configs)
    with open(data_file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8)
        data = data.astype(np.int16)
        pprint(data)
        channel1,channel2 = conv_format212(data)
        channel1 -= data_configs['channel1']['zero_offset']
        channel1 /= data_configs['channel1']['gain']
        channel2 -= data_configs['channel2']['zero_offset']
        channel2 /= data_configs['channel2']['gain']
        return channel1,channel2
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
```

