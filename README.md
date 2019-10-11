# A_Little_fog_for_a_Large_Turn
Code for the WACV paper, "A Little Fog for a Large Turn"

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
-  Linux or macOS
-  Python 3
-  Preferably NVIDIA GPU + CUDA CuDNN
-  Pytorch >= 1.0

Note: For DistanceGAN code: Python2 is used by the original authors instead and we continue to use the same.


## Citation




## Acknowledegements
-  CycleGAN repository
-  DistanceGAN repository
-  AutoPilot by SullyChen
-  Comma AI 

