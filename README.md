# EE8227
Effects of Adversarial Training and Adversarial Fine-Tuning U-Net for Semantic Segmentation

Dataset 1 can be downloaded from https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Dataset 2 and 3 can be downloaded from https://www.kaggle.com/code/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen

Step 1: Download all datasets 

Step 2: Run "TrainUnet.py" 2 times using Dataset 1. This will produce models 1 and 4. You must change path files in lines 135, 136, 253, 270, 271. Change model names in line 251 and 252 accordingly. The file is currently set up for training on clean images. If you want to train on adversarial examples then uncomment lines 247-248, 262-267 and comment out line 256.

Step 3. Run "TuneUnet.py" 4 times using Dataset 1. This will produce models 2, 3, 5 and 6. You must change path files in lines 128, 129, 155, 156, 183, 193, 209 and 210. Change model names in line 182, 183, 191 and 192 accordingly. The file is currently set up for training on adversarial images. If you want to train on clean examples then uncomment line 206 and comment out lines 198-204. 

Step 4. Run "Test.py". This will produce IoU test metrics for all models. You must change path files in lines 114, 115, 154, 161, 167, 173, 179, 185 and 192. This file is currently set up for testing with clean images. To test on FGSM attacks uncomment lines 47, 48 and 138. To test on PGD attacks uncomment line 137.
