MAF validation
Original repo: https://github.com/adityabingi/DCGAN-TF2.0

## Install (MAF with TF2.9)
```
pip install numpy scipy tqdm matplotlib sklearn scikit-learn
python dcgan.py --train 2>&1 | tee dcgan_tf2_maf-12jan.log
```

## Prepare data (celebA)
Note: Google drive quota is not so big, so the file usually can't be downloaded by `gdown`, if so please download manually and upload to server
```
pip install gdown
gdown --fuzzy https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=share_link&resourcekey=0-dYn9z10tMJOBAkviAcfdyQ
unzip img_align_celeba
```

## Training
Note:
 - Change the data dir (4th line) in the config.py to the one you extracted
    ```
    data_dir_path = 'CelebA/img_align_celeba/' --> data_dir_path = '/data/work/dataset/img_align_celeba/'
    ```
 - First is to fix the deprecated function of Tensorflow 2.0.0 Find and Replace all `experimental_run_v2` with `run` in dcgan.py
 - Edit code to create tfrecord before training start:
    ```
    # import prepare_tfrecords function from dataset
    from dataset import prepare_dataset, prepare_tfrecords

    # line 184: uncomment prepare_tfrecords func on first run
    prepare_tfrecords()
    ```

## Training
```
python dcgan.py --train 2>&1 | tee dcgan_tf2_maf_v23.1.1.log
```
-----------
# Original README
-------------
# DCGAN

Minimalistic tf 2.0 implementation of DCGAN with support for distributed training on multiple GPUs.

This work is aimed to generate novel face images similar to CelebA image dataset using Deep Convolutional Generative Adversarial Networks (DCGAN).

For theory of GAN's and DCGAN refer these works:
1. [Generative Adversarial Nets](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
2. [NIPS 2016 Tutorial:Generative Adversarial Networks](https://arxiv.org/pdf/1701.00160.pdf)
3. [UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS](https://arxiv.org/pdf/1511.06434.pdf)

Code compatibility:
python>=3.6
Tensorflow==2.0.0

## Dataset

`python download_celebA.py 0B7EVK8r0v71pZjFTYXZWM3FlRnM CelebA.zip`

Extract CelebA.zip and the images are found in the img_align_celeba folder.

Data Processing:

All the images in the celeba dataset are of (218 ,178, 3) resolution and for this work all the images are cropped by carefully choosing the common face region (128, 128, 3) in all the images. Check data_crop in config.py

## Usage

For multi-gpu training:

`python dcgan.py --train`

To run on single GPU run the above code by replacing strategy = tf.distribute.MirroredStrategy(devices) (line 185 in dcgan.py) with 
strategy=tf.distribute.OneDeviceStrategy(device='/GPU:0') and configure device variable by passing it with specific gpu id like "/device:GPU:0" or "/device:GPU:1" 

You can also run single GPU training with tf.distribute.MirroredStrategy also by simply setting num_gpu = 1 in config.py.

For Generating new samples:

`python dcgan.py --generate`


## Results

Following are the results after training GAN on 128x128 resolution CelebA face images for 15 epochs on 2 NVIDIA Tesla K80 GPUs with global batch size of 32 (batch size 16 per gpu). Detailed configuration can be found in config.py 

Fake images generation during course of GAN training:

![training-result](results/dcgan_training.gif)

Fake Images Generation after 15 Epochs:
![results_15epoch](results/fakes_epoch15.jpg)

