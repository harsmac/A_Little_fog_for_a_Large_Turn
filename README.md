# A Little Fog for a Large_Turn
Code for the WACV 2020 paper, "A Little Fog for a Large Turn" <br>
Arxiv paper link : [https://arxiv.org/abs/2001.05873](https://arxiv.org/abs/2001.05873)
Website link : [https://code-assasin.github.io/little_fog/](https://code-assasin.github.io/little_fog/)

## Abstract
Small, carefully crafted perturbations called adversarial perturbations can easily fool neural networks.
However, these perturbations are largely additive and not naturally found. We turn our attention to the field of Autonomous navigation wherein adverse weather conditions such as fog have a drastic effect on the predictions of these systems. These weather conditions are capable of acting like natural adversaries that can help in testing models.
To this end, we introduce a general notion of adversarial perturbations, which can be created using generative models and provide a methodology inspired by Cycle-Consistent Generative Adversarial Networks to generate adversarial weather conditions for a given image.
Our formulation and results show that these images provide a suitable testbed for steering models used in Autonomous navigation models. Our work also presents a more natural and general definition of Adversarial perturbations based on Perceptual Similarity.

## Examples
Some sample Deviations seen in popular steering angle predictors due to fog : 
![alt text](https://github.com/code-Assasin/A_Little_fog_for_a_Large_Turn/blob/master/images_readme/stack.png "Samples")

Fooling Models: Ground truth Steering Angle (in radians) for each of the original test samples. The angles right below indicate the ordered pair of predicted steering angle by AutoPilot and Comma AI respectively. From the second row onward, we indicate the image translation model used and respective steering model it was trained on. The angle below each of those images indicates the prediction by the steering model for the generated foggy image.

## Prerequisites 
```
-  Linux or macOS
-  Python 3
-  Preferably NVIDIA GPU + CUDA CuDNN
-  Pytorch>=0.4.1
-  torchvision>=0.2.1
-  dominate>=2.3.1
-  visdom>=0.1.8.3
```
Note: For DistanceGAN code: Python2 is used by the original authors instead and we continue to use the same.


## Usage

### CycleGAN : 
In the file "cycle_gan_code/models/cycle_gan_fool_model.py" update the steering model to be attacked and the alpha and theta values. Then come back to the CycleGAN folder (cycle_gan_code) and run: 
```
python3 train.py --dataroot /gan_train/ --name model_name --model cycle_gan_fool --preprocess none --batch_size 40 --gpu_ids 0,1,2,3 --norm instance --init_type kaiming --display_port 8097 --no_dropout --lambda_identity 3 --niter_decay 500 --niter 100
```
Dataroot should have following subfolders: ```trainA trainB testA testB```

### DistanceGAN:
In the file "cycle_gan_code/models/cycle_gan_fool_model.py" update the steering model to be attacked and the alpha and theta values. Then come back to the CycleGAN folder (cycle_gan_code) and run: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataroot /gan_train/ --name model_name --model distance_gan --loadSize 128 --fineSize 128 --batchSize 6 --norm instance --nThreads 8 --use_cycle_loss --max_items 5 --identity 3 --gpu_ids 1
```

## Pretrained models: 
Coming soon!!


## Citation
If you use this work for your research, please cite our paper: 
```
@misc{machiraju2020little,
    title={A Little Fog for a Large Turn},
    author={Harshitha Machiraju and Vineeth N Balasubramanian},
    year={2020},
    eprint={2001.05873},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Acknowledegements
Our code is inspired by the following repositories: 
-  CycleGAN repository [link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
-  DistanceGAN repository [link](https://github.com/sagiebenaim/DistanceGAN)
-  AutoPilot by SullyChen [link](https://github.com/SullyChen/Autopilot-TensorFlow)
-  Comma AI [link](https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/rambo/README.md)

