# transformer_CHN2ENG
使用transformer模型实现机器翻译任务，针对中译英的翻译任务
## 数据文件
cn-eng.txt中包含90000条中英文句对。
## 安装环境
假设你的系统(适用于Linux和MacOS)已经安装好Anaconda：
```
# 首先在终端中进入当前目录

# 执行下方命令以创建运行项目的conda环境，可将下面环境名称myenv更换为其他名称
conda create --name myenv python=3.7.3
source activate myenv
pip install -r requirements.txt
```
## 使用方法
### 运行训练
```
# 由于训练过程日志使用wandb记录，因此需要首先在配置文件 c2e_configs.yaml 中填入你自己的 wandb_entity

# 执行以下命令在linux系统中后台运行训练脚本
nohup python3 -u train_model.py > console.log 2>&1 & # console.log用以保存打印台输出

# 使用下方命令，可以实时观察打印台输出
tail -f console.log
```
训练中间loss最优的模型以pt格式保存在models/intermediate中，最终模型保存在目录models中，你可以通过修改 c2e_configs.yaml 文件尝试在不同参数下训练模型。

### 推理
```
# 可在命令参数中指定用以推理的模型的名称和路径
python make_inference.py [--model_path MODEL_PATH]
                         [--input_lang_path INPUT_LANG_PATH]
                         [--output_lang_path OUTPUT_LANG_PATH]
                         [--device DEVICE] 
                         
# 如果只需默认参数 可以忽略后面的命令参数 直接运行 python make_inference.py
# input_lang.pkl 文件可用于将输入的源文本token和数值token的互相转换
# output_lang.pkl 文件可用于将输出的目标文本token和数值token的互相转换
# device 默认值为auto 自动根据系统情况选择cuda还是cpu推理 另外支持指定 'cpu' 或 'cuda
```
执行上述命令后，你可在控制台输入你想尝试翻译的中文句子，按回车键可等待10s左右查看翻译结果，单次运行允许10条翻译。可以重复运行。
