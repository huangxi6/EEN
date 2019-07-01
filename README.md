# EEN

This is an efficient implementation of EEN (Multi-class Human Body Parsing with Edge-Enhancement Network). The code is based upon [this implementation](https://github.com/liutinglt/CE2P).

### Download

Plesae download [LIP and CIHP](http://sysu-hcp.net/lip/overview.php) dataset.

Pascal-Person-Part dataset and trained models can be found at [baidu drive](https://pan.baidu.com/s/1nZImrFhtBLylFum3TmAoUg) (the password is '7cas') or [google drive](https://drive.google.com/open?id=1MDcTbIjA5XgP_tnrAN9yR3q41LObljFC).


### Environments

+ Python 3.5   

+ PyTorch 0.4.1  

+ cffi

+ matplotlib

+ numpy        

+ opencv-python

+ scipy

+ tqdm

+ You need to use InPlace-ABN with CUDA implementation, which must be compiled with the following commands:

```bash
cd libs
sh build.sh
python build.py
``` 
+ The model is trained on two NVIDIA TITAN RTX 2080 Ti GPUs. It will take up about 16G.


### Training

+ Please set the dataset dir in file 'run.sh'. The contents of each dataset include: 

  ─ train_images   

  ─ train_labels  

  ─ val_images  

  ─ val_labels    

  ─ train_id.txt  

  ─ val_id.txt  

+ Please put the pretrained resnet101-imagenet.pth in './dataset/'.

+ Run the `sh run.sh`. 

### Evaluation

If you want to evaluate the trained models on LIP and CIHP, you can run the 'sh run_evaluate.sh'.

