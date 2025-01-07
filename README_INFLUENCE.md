
ubuntu 20

miniconda (随意)

conda info --env 

conda create -n im2rc python=3.8

conda activate im2rc

pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

pipip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple