# mixup_augmentation
implementation mixup data augmentation with numpy and keras

通常のmixupは2つのサンプルを混合する。
この実装では2つ以上のサンプルを混合する。
そのため、混合比はディリクレ分布からサンプリングされる。
The typical mixup mixes two samples.
In this implemention, mixup mixes two or more samples.
Therefore, the mixing ratio is sampled from the Dirichlet distribution.


# mixup.py

# class MixupGenerator
numpyで実装されたmixupのジェネレーター
Generator of mixup implemented with numpy

# class MixupSequence
kerasのSequenceを使ったmixupのジェネレーター
Generator of mixup using Sequence of keras

# class ImageMixupSequence
kerasのimageDataAugmentationとmixupを組み合わせたジェネレーター
Generator combining keras's imageDataAugmentation and mixup

#　mixup_test.py
Verification of accuracy with cifar10 using resnet.
I reffered to keras tutorial https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py.


