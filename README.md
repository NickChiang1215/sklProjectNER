
### 訓練：
1. 將訓練資料放到 datasets/sklJoint/
2. pretreained model放到pretrained_model/chinese_roberta_wwm_ext_pytorch
3. 執行 run_joint.sh

### Inference:
1. 將訓練好的模型放到skl_model資料夾下。目前模型：https://drive.google.com/drive/u/2/folders/1z5UZiF0KDITYZoGZ_CHDASCfl_sjRVLR
2. docker-compose建立環境及執行

### 訓練資料格式
```text
美	B-LOC
国	I-LOC
的	O
华	B-PER
莱	I-PER
士	I-PER
我	O
跟	O
他	O
```
