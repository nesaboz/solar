conda init zsh

conda update -n base -c conda-forge conda

# apt-get is for Linux, use `brew` on Mac
apt-get update && apt-get install libgl1 -y  # avoids ImportError: libGL.so.1: cannot open shared object file: No such file or directory

# to get IP address
apt install curl
curl ifconfig.me

# to set up environment in home since it might get deleted otherwise
repo_name=home/solar  #
yes | conda create --prefix $repo_name python=3.10
conda activate $repo_name

pip install -U torch numpy pandas matplotlib torchviz scikit-learn tensorboard torchvision tqdm torch-lr-finder ipyplot ipywidgets opencv-python torchmetrics
yes | conda install -c conda-forge jupyter_contrib_nbextensions graphviz python-graphviz

# to install environment as kernel, rope_name can't have /
repo_name=${repo_name//[\/]/_}
ipython kernel install --name $repo_name --user

# clone repo
git clone https://github.com/nesaboz/solar.git

