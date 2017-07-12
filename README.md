### Tips:
- TTA (test time augmentation): combine 12 predictions including crop_{}, crop_{}_rotate_90, crop_{}_rotate_180, crop_{}flip, crop{}_rotate_90_flip, crop_{}_rotate_180_flip, where {} is replaced by 224 or 256. OR 10-crop-224 method (used in ImageNet)
- Input image size: 256,288,224
- Trainging models with 224 (center crop) input rather than 256 (maybe because AdaptivePooling)
- Different scale: use 256,288,224 for training and 320 for testing (improved results really nicely)
- Random scale: random scale for training (model see images at different scales), same for testing.
- Train-val split: 10 / 20-fold
- Optimisation
    - sgd with step learning rates, e.g. 0.01, 0.001, 0.0001
        - 0-10: 0.001
        - 10-20: 0.0001
        - 20-25: 0.00001
    - use train/validation to determine stopping criteria and when to change learning rates
- Pretrain models (https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975)
    - resnet34 easiest to work with and fastest to train. the tricks apply to all networks.
    - VGG16 structure
    - Dense121 can get 0.93 alone (with TTA)
- Batch size
- Use mean and std from Image net. I'm using it and it works nice.

### Pretrained Models:
- CNN Finetune: https://github.com/flyyufelix/cnn_finetune
- DenseNet: https://github.com/flyyufelix/DenseNet-Keras
- ResNet-101: https://gist.github.com/flyyufelix/65018873f8cb2bbe95f429c474aa1294
- ResNet-152: https://gist.github.com/flyyufelix/7e2eafb149f72f4d38dd661882c554a6
- SqueezeNet: https://github.com/rcmalli/keras-squeezenet
- Inception v4: https://github.com/titu1994/Inception-v4/releases
- VGG16: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
- VGG19: https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d
- Other Keras models: https://keras.io/applications/