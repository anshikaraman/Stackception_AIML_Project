## TEAM STACKCEPTION\
**Members:**\
Anshika Raman       210050014\
Komma Sharanya      210050086\
Namrata Jha         210050104\
Shriyank Tatawat    210050147\
Vaibhav Vishal      210050159

Make sure following tree structure is maintained\
Add birds and CUB dataset folder to input

STACK_GAN_Project\
├── cfg\
│   ├── s1.yml\
│   └── s2.yml\
├── output\
│   ├── log\
│   ├── model_stage1\
│   ├── model_stage2\
│   ├── results_stage1\
│   └── results_stage2\
├── input\
│   └── data\
│       ├── birds\
│       └── CUB_200_2011\ 
│           ├── (200 folders)\
│                ├── (11,788 images)\
└── src\
    ├── args.py\
    ├── dataset.py\
    ├── engine.py\
    ├── layers.py\
    ├── train.py\
    └── util.py\

Parameters/variables to be changed before running the training/testing:\

- cfg/s1.yml : update the arguments in this file before running stage1\

- args.py   :  update the 'PATH_STR' variable to contain the path of the 'STACK_GAN_Project' directory\

- python3 src/train.py --conf cfg/s1.yml : for training stage1\

- cfg/s2.yml : update the arguments in this file before running stage2\

- python3 src/train.py --conf cfg/s2.yml : for training stage2\

- for creating inferences from model, comment run(args_) in main of train.py and uncomment the commented part below which calls the sample function\
