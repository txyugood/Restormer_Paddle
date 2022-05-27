# 基于Paddle复现《Restormer: Efficient Transformer for High-Resolution Image Restoration》
## 1.简介
由于CNN在从大规模数据中学习广义图像先验知识方面表现良好，这些模型已被广泛应用于图像恢复等相关任务。最近，另一类神经结构Transformers在自然语言和High-Level视觉任务上显示出显著的性能提升。虽然Transformer模型缓解了CNN的不足（即有限的感受野和对输入内容的适应性），但其计算复杂度随空间分辨率呈二次增长，因此无法应用于大多数涉及高分辨率图像的图像恢复任务。在这项工作中，我们提出了一种高效的转换器模型，通过在构建模块（多头注意和前馈网络）中进行几个关键设计，它可以捕获长距离的像素交互，同时仍然适用于大型图像。我们的模型名为RestorationTransformer（Restormer），在多个图像恢复任务上实现了SOTA的结果，本Repo主要复现了图像去噪的模型。

原repo: [https://github.com/swz30/Restormer](https://github.com/swz30/Restormer)

论文地址: [https://arxiv.org/pdf/2111.09881.pdf?ref=https://githubhelp.com](https://arxiv.org/pdf/2111.09881.pdf?ref=https://githubhelp.com)


## 2.复现精度
原repo采用的是8卡训练，这里我改为4卡，同时iters 乘以2，学习率除以2。
在CBSD68测试集的测试效果如下表,达到验收指标,PSNR: 34.39。

| Network | opt | iters | learning rate | batch_size | dataset | GPUS | PSNR  |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| Restormer | AdamW  | 600000 | 1.5e-4 | 8  |CBSD68 | 4 | 34.39 |


## 3.数据集

下载地址:

[https://aistudio.baidu.com/aistudio/datasetdetail/140244](https://aistudio.baidu.com/aistudio/datasetdetail/140244)

解压数据集
```shell
cat DFWB.tar.gza* | tar zxv
```

最优权重:

链接: [https://pan.baidu.com/s/14lxC6gHrr6BXHJBZgY1C_g](https://pan.baidu.com/s/14lxC6gHrr6BXHJBZgY1C_g)

提取码: t067 

## 4.环境依赖
PaddlePaddle == 2.2.0

scikit-image == 0.19.2

## 5.快速开始

### 模型训练

训练至少需要4卡资源，配置默认为4卡，如需8卡训练可修改configs/GaussianColorDenoising_Restormer.yml文件。将其中跟iters相关的数值除以2，同时将学习率相关数值乘以2.
多卡训练，启动方式如下：
```shell
python -u -m paddle.distributed.launch / train.py -opt configs/GaussianColorDenoising_Restormer.yml 
```
多卡恢复训练，启动方式如下：
```shell
python -u -m paddle.distributed.launch / train.py -opt configs/GaussianColorDenoising_Restormer.yml --resume ../245_model
```

参数介绍：

opt: 配置路径

resume: 从哪个模型开始恢复训练，需要pdparams和pdopt文件。


### 模型验证

除了可以再训练过程中验证模型精度，还可以是val.py脚本加载模型验证精度，执行以下命令。

```shell
python val.py --input_dir ../data/Datasets/test/ --weights best_model.pdparams --model_type blind --sigmas 15 
```

输出如下：

```shell
/home/aistudio/Restormer_Paddle
Compute results for noise level 15
W0527 09:34:27.776396  2557 gpu_context.cc:278] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 10.1
W0527 09:34:27.780679  2557 gpu_context.cc:306] device: 0, cuDNN Version: 7.6.
Loading pretrained model from ../best_model.pdparams
There are 406/406 variables loaded into Restormer.
===>Testing using weights:  ../best_model.pdparams
------------------------------------------------
[Eval] PSNR: 34.39220432429305
```

### 单张图片预测
本项目提供了单张图片的预测脚本，可根据输入图片生成噪声，然后对图片进行降噪。会在result_dir指定的目录下生成denoise_0000.png和noise_0000.png两张图片。使用方法如下：
```shell
python predict.py --input_images ../data/CBSD68/0000.png \
--weights best_model.pdparams \
--model_type blind --sigmas 15 --result_dir ./output/
```

参数说明：

input_images:需要预测的图片

weights: 模型路径

result_dir: 输出图片保存路径

model_type: 模型类型，本项目只训练了blind模式。

sigmas: 噪声等级。


在噪声等级15下的预测样例:


 <center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/0002.png width = "30%" alt=""/>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/noise_0002.png width = "30%" alt=""/>
        <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/denoise_0002.png width = "30%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      从左到右分别是clear、nosie、denoise
  	</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/0000.png width = "30%" alt=""/>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/noise_0000.png width = "30%" alt=""/>
        <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src=demo/denoise_0000.png width = "30%" alt=""/>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
      从左到右分别是clear、nosie、denoise
  	</div>
</center>



### 模型导出
模型导出可执行以下命令：

```shell
python export_model.py -opt ./test_tipc/configs/GaussianColorDenoising_Restormer.yml --model_path ./output/model/last_model.pdparams --save_dir ./test_tipc/output/
```

参数说明：

opt: 模型配置路径

model_path: 模型路径

save_dir: 输出图片保存路径

### Inference推理

可使用以下命令进行模型推理。该脚本依赖auto_log, 请参考下面TIPC部分先安装auto_log。infer命令运行如下：

```shell
python infer.py
--use_gpu=False --enable_mkldnn=False --cpu_threads=2 --model_file=./test_tipc/output/model.pdmodel --batch_size=2 --input_file=test_tipc/data/CBSD68 --enable_benchmark=True --precision=fp32 --params_file=./test_tipc/output/model.pdiparams 
```

参数说明:

use_gpu:是否使用GPU

enable_mkldnn:是否使用mkldnn

cpu_threads: cpu线程数
 
model_file: 模型路径

batch_size: 批次大小

input_file: 输入文件路径

enable_benchmark: 是否开启benchmark

precision: 运算精度

params_file: 模型权重文件，由export_model.py脚本导出。



### TIPC基础链条测试

该部分依赖auto_log，需要进行安装，安装方式如下：

auto_log的详细介绍参考[https://github.com/LDOUBLEV/AutoLog](https://github.com/LDOUBLEV/AutoLog)。

```shell
git clone https://gitee.com/Double_V/AutoLog
cd AutoLog/
pip3 install -r requirements.txt
python3 setup.py bdist_wheel
pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```


```shell
bash test_tipc/prepare.sh ./test_tipc/configs/Restormer/train_infer_python.txt 'lite_train_lite_infer'

bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/Restormer/train_infer_python.txt 'lite_train_lite_infer'
```

测试结果如截图所示：

<img src=./test_tipc/data/tipc_result.png></img>

## 6.代码结构与详细说明

```
Restormer_Paddle
├── README.md # 说明文件
├── logs # 训练日志
├── configs # 配置文件
├── data # 数据变换
├── dataset.py # 数据集路径
├── demo # 样例图片
├── export_model.py # 模型导出
├── infer.py  # 推理预测
├── metrics  # 指标计算方法
├── models # 网络模型
├── predict.py # 图像预测
├── test_tipc # TIPC测试链条
├── train.py # 训练脚本
├── utils # 工具类
└── val.py # 评估脚本

```

## 7.模型信息

| 信息 | 描述 |
| --- | --- |
|模型名称| Restormer |
|框架版本| PaddlePaddle==2.2.0|
|应用场景| 降噪 |
