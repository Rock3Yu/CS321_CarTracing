# CS321_CarTracing

maddpg@10.16.29.94

git fetch

git branch -a

git push origin master:main

git pull origin main:master

scp maddpg@10.16.29.94:v1/yu/test.py D:\Downloads\SHARE

# to train a model
You should carefully set the super paremeter before every experiment

nohup python -u main.py --overwrite True >test.log 2>&1 & 

tensorboard --logdir ./log/tensorboard/spread_l4/ --port 6789