# coding=gbk
import os
import wave
import numpy as np
import pandas as pd
from pydub import AudioSegment
from SpeakerDirization import audioSegmentation as aS

import matplotlib.pyplot as plt



path = r"" + os.getcwd()
print(path)
files = os.listdir(path)
files = [path + "\\" + f for f in files if f.endswith('.wav')]


def CutFile():
    for i in range(len(files)):
        aS.speakerDiarization(r"" + files[i], 3)
        print("CutFile File Name is ", files[i])
        f = wave.open(r"" + files[i], "rb")
        params = f.getparams()
        print(params)
        nchannels, sampwidth, framerate, nframes = params[:4]
        # 读取格式信息
        # 一次性返回所有的WAV文件的格式信息，它返回的是一个组元(tuple)：声道数, 量化位数（byte    单位）, 采
        # 样频率, 采样点数, 压缩类型, 压缩类型的描述。wave模块只支持非压缩的数据，因此可以忽略最后两个信息
        f.close()

        #输入标记数据
        hola = files[i].replace('.wav', '.xlsx')
        # my_matrix = np.loadtxt(open(hola, "rb"), delimiter=",", skiprows=0)
        my_matrix = pd.read_excel(hola, usecols=[1])
        Data = my_matrix.values
        Data.shape = -1, 1
        # print(Data)

        #分割重组
        CutTimeDef = 0.2  # 以0.2s截断文件
        cut_time = 1000 * CutTimeDef
        n_max = nframes / framerate *5
        silence = AudioSegment.silent(duration=cut_time)
        for j in range(3):
            FileName = r"" + files[i][0:-4] + "-" + str(j + 1) + ".wav"
            print(FileName)
            audio = AudioSegment.from_wav(files[i])
            new_audio = 0
            n = 0
            while n < n_max - 3:

                if Data[n] == j:
                    new_audio = new_audio + audio[n*cut_time:(n+1)*cut_time]
                else:
                    new_audio = new_audio + silence
                n = n + 1
            new_audio.export(FileName, format="wav")


if __name__ == '__main__':
    CutFile()

    print("Run Over")