## Introduction ##

This repository is about paper SpeakerGAN , and is implemented by Mingming Huang (dyyzhmm@163.com), Tiezheng Wang (wtz920729@163.com) and thanks for advice from TongFeng.

## SpeakerGAN paper ##

SpeakerGAN: [Speaker identification with conditional generative adversarial network](https://www.researchgate.net/publication/344057150_SpeakerGAN_Speaker_Identification_with_Conditional_Generative_Adversarial_Network)， by Liyang Chen , Yifeng Liu , Wendong Xiao , Yingxue Wang ,Haiyong Xie.

## Usage ##
For train / test / generate: 

	python speakergan.py
    
You may need to change the path of wav vad preprocessed files.
    
## Our results ##

	acc: 94.27% with random sampled testset. 

	acc: 93.21% with fixed start sampled testset.

	using model file: model/49_D.pkl
    
    acc: 98.44% on training classification accuracy with real samples.

There is about **4% gap on testset** lower compared to paper result.    We can't find out the reason. **We want your help !**

![Alt accuracy](logs/acc.png)    
![Alt loss_d_loss_g](logs/loss.png)   
![Alt learning_rate](logs/lr.png)   
    
## Details of paper ##

The following are details about this paper.

================ input ==================

1. feature: fbank, 8000hz, 25ms frame, 10ms overlap. shape:(160,64)

2. dataset: librispeech-100 train-clean-100  POI:251

3. data preprocess:  vad、mean and variance normalization, shuffled.

4. 60% train. 40% test.


================ model architecture ==================

1. dataflow: data -> feature extraction -> G & D

2. model architecture:

      G: gated CNN, encoder-decoder, Huber loss + adversarial loss
   
      D: ResnetBlocks, template average pooling, FC, softmax, crossentropy loss + adversarial loss

3. G: shuffler layer, GLU

4. D: ReLU


================ training ==================

1. lr: 0-9, 0.0005 | 9-49, 0.0002

2. L(d): λ1 λ2 = 1

3. batch_size: 64

4. D_train steps / G_train steps = 4

5. Ladv Loss: Label smoothing, 1 -> 0.7 ~ 1.0, 0 -> 0 ~ 0.3


======== not sure or differences with paper ========

1. weights,bias initialize function, use: xavier_uniform and zeros

2. pytorch huber_loss： + 0.5 to be same with paper.  but no implement here.

3. for shorter wav, paper: padded with zero. we: padded with feature again.

4. gated cnn architecture.

5. we use webrtcvad mode(3) for vad preprocess.



