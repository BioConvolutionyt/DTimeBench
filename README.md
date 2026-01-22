# DTimebench

一个用于**入门学习/复现**的深度学习时间序列预测（回归）小项目，包含多种基础模型结构及对应的可运行示例脚本。

## 内容概览

- **基础时序模型/骨干**
  - `TCN.py`：TCN（Temporal Convolutional Network）
  - `CNN_LSTM_or_BiLSTM.py`：CNN-LSTM / BiLSTM
  - `CNN_GRU_or_BiGRU.py`：CNN-GRU / BiGRU
- **注意力/增强模块**
  - `Self_Attention.py`：Self-Attention（用于序列特征）
  - `SE_or_ECA_or_CBAM.py`：SE / ECA / CBAM（通道/空间注意力变体）
- **示例（可直接运行）**
  - `TCN-GRU-ECA demo.py`：TCN + GRU + ECA 的端到端训练/评估示例
  - `CNN-LSTM-Self_Attention demo.py`：CNN + LSTM + Self-Attention 的端到端训练/评估示例

说明：部分在 PyTorch 中已有内置实现的模型/层（如 `GRU`、`LSTM`）不在本项目中重复实现，示例脚本中直接调用 `torch.nn` 的对应接口。

## 数据集

使用 `weather.csv`（weather 数据集），包含 **21 个气象指标**（如气温、湿度等）。数据在 **2020 年**按 **10 分钟**间隔记录。示例脚本基于滑动窗口构造序列样本，并对目标进行一步预测。

## 运行方式

在项目根目录安装依赖后，直接运行示例脚本即可：

```bash
python "TCN-GRU-ECA demo.py"
python "CNN-LSTM-Self_Attention demo.py"
```

依赖包括：`torch`、`numpy`、`pandas`、`scikit-learn`、`matplotlib`。


