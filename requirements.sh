conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c pytorch -c nvidia faiss-gpu
conda install -y transformers
conda install -y pandas
pip install --upgrade transformers
pip install sentence_transformers
pip install sentencepiece
pip install sacremoses
pip install langdetect
pip install accelerate
git clone https://github.com/bofenghuang/vigogne.git
cd vigogne
pip install .
pip install pika
pip install pymongo