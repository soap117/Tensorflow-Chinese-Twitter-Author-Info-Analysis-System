# Tensorflow Chinese Twitter(微博) Author Info Analysis System
## Motivation
It's very easy for a man with proper common sense and social experience to guess a speaker's social backgrouds according to its' dialogue.
If machine can do the same thing it should be very interesting and has a potential commercial value. So design and trained this system based on the data collected by a web spider. 
## Method
### Hybrid bone network
RNN CNN, Doc2vec three approches are used to extract embedding vector of a short text. And final a nerual classifier is applied to give out the prediction.
## How to use it
create 5 directories under the branch: favor_data, favorite, general_data, togather, save256. Put your favorite songs in favorite in wav form, then put general songs(including your favorite ones) in togather. <br/>
run data_prepare.py to generate data<br/>
run train.py to train the model<br/>
run data_emb.py to generate the embeddings<br/>
run cluster.py to estimate the cluster centers and threshold<br/>
run eval.py  _path_of_wav_file_  _threshold_  to test if a new song will be accepted

