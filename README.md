# OCTraN

Modern approaches for vision-centric environment perception for autonomous navigation make extensive use of self-supervised monocular depth estimation algorithms to predict the depth of a 3D scene using only a single camera image. However, when this depth map is projected onto 3D space, the errors in disparity is magnified,  resulting in a depth estimation error that increases quadratically as the distance from the camera increases. Though Light Detection and Ranging (LiDAR) can solve this issue, it is expensive and not feasible for many applications. To address the challenge of accurate ranging with low-cost sensors, we propose a transformer architecture that uses iterative-attention to convert 2D image features into 3D occupancy features and makes use of convolution and transpose convolution to efficiently operate on spatial information. We also develop a self-supervised training pipeline to generalize the model to any scene by eliminating the need for LiDAR ground truth by substituting it with pseudo-ground truth labels obtained from neural radiance fields (NeRFs) and boosting monocular depth estimation.


# Dataset

 - Raw dataset link - https://drive.google.com/drive/folders/1ouZk8stDobJtpDvYKBPKg0bWNR6bIX-U?usp=sharing
 - Depth Dataset link - https://drive.google.com/drive/folders/1pzDVOu_Kpo_1sSdOQvF3n7zZ44NBLdzN?usp=share_link

# Acknowledgements
Our work builds on and uses code from <a src="https://github.com/lucidrains/perceiver-pytorch/">Perceiver</a> and <a src="https://github.com/lucidrains/vit-pytorch/">ViT</a>. We'd like to thank the authors for making these libraries available.
