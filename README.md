# Team Stackception
Members:\
Anshika Raman       210050014 \
Komma Sharanya      210050086 \
Namrata Jha         210050104 \
Shriyank Tatawat    210050147 \
Vaibhav Vishal      210050159 \

```bash
pip3 install -r requirements.txt 
```
- Download the The Caltech-UCSD Birds-200-2011 (CUB) Dataset from: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html and extract it in `input/data/` to create `input/data/CUB_200_2011` directory

- Download the text descriptions from: https://tinyurl.com/yz8h2s2b and extract it in `input/data/` to create `input/data/birds` directory

Make sure following tree structure is maintained\
Add birds and CUB dataset folder to `input/data`
Download the dataset from here:\


```
Stackception_AIML_Project
│   README.md   
│	requirements.txt
│
└──>cfg
│   │   s1.yml
|   |   s2.yml
│   
└──>input
│   │
│   │
│   └──>data
│   |   │
│   |   └──>birds
│   |   |    
|   |   └──>CUB_200_2011
|
|
└──>model_stage1
|
└──>model_stage2
|
└──>src
│   │   args.py
|   |   dataset.py
│   │   engine.py
│   │   environment.yml
│   │   layers.py
│   │   train.py
│   │   util.py
```

Parameters/variables to be changed before running the training/testing:

- args.py   :  update the 'PATH_STR' variable to contain the path of the 'STACK_GAN_Project' directory

- cfg/s1.yml : update the arguments in this file before running stage1

- python3 src/train.py --conf cfg/s1.yml : for training stage1

- args.py   :  update the 'PATH_STR' variable to contain the path of the 'STACK_GAN_Project' directory

- cfg/s2.yml : update the arguments in this file before running stage2

- python3 src/train.py --conf cfg/s1.yml : for training stage1

- python3 src/train.py --conf cfg/s2.yml : for training stage2

- cfg/s2.yml : update the arguments in this file before running stage2

- for creating inferences from model, comment run(args_) in main of train.py and uncomment the commented part below which calls the sample function

- python3 src/train.py --conf cfg/s2.yml : for training stage2

- for creating inferences from model, comment run(args_) in main of train.py and uncomment the commented part below which calls the sample function and change embedding path

--------------------------------------------------------------------------------------------

# References:
Paper1 : https://arxiv.org/pdf/1612.03242.pdf \
Paper2 : https://arxiv.org/pdf/1605.05396.pdf
