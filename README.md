# SpeakerDirization

## 任务

本项目能够从需要能够从多人语音文件中分离出多个对话文件，这些文件保持原有的时间轴并且只保留其中一个说话者的声音片段。

## 安装与使用

```bash
# install
pip install numpy pandas pydub matplotlib
git clone https://github.com/Litosty/SpeakerDirization
# use
python SpeakerD.py .\demo\demo.wav
```



## 参考文献

1. Quan Wang, Carlton Downey, Li Wan, Philip Andrew Mans- field, and Ignacio Lopz Moreno, “Speaker diarization with lstm,” in International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018, pp. 5239–5243.

2. Zhang A , Wang Q . FULLY SUPERVISED SPEAKER DIARIZATION[J]. 2018. 

3. 李芝峰. 基于深度学习的多说话人语音分离技术研究[D].辽宁大学,2019.

4. 朱唯鑫. 多人对话场景下的说话人分割聚类研究[D].中国科学技术大学,2017.

5. Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding：https://github.com/pyannote/pyannote-audio

6. Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications：https://github.com/tyiannak/pyAudioAnalysis