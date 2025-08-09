conda create -n tinyllava_factory_cu18 python=3.10 -y
conda activate tinyllava_factory_cu18
pip install --upgrade pip  # enable PEP 660 support

conda install -c nvidia cuda-toolkit=11.8  # or 11.8 if that’s your stack
# conda install -c nvidia cuda-toolkit=12.1  # or 11.8 if that’s your stack
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidi
# conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# pip install flash-attn --no-build-isolation
pip install -e .
pip install flash-attn==2.5.7 --no-build-isolation

# pip install -U "deepspeed>=0.17.2"
# pip uninstall -y scikit-learn numpy
# pip install scikit-learn numpy