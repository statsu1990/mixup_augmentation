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

# mixup_test.py
Verification of accuracy with cifar10 using resnet.  
I reffered to keras tutorial https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py.

# results
 image_augmantation	mix_num	alpha	epoch	Train loss	Train accuracy	Test loss	Test accuracy  
resnet20(v2)	-	-	-	200	0.089	1.000	1.051	0.842	  
resnet20(v2)	-	2	0.05	200	0.100	1.000	0.650	0.840	  
resnet20(v2)	use	-	-	200	0.227	0.967	0.421	0.912	  
resnet20(v2)	use	-	-	400	0.220	0.956	0.415	0.912	  
resnet20(v2)	use	2	0.05	200	0.226	0.962	0.387	0.910	  
resnet20(v2)	use	2	0.2	200	0.250	0.954	0.368	0.914	  
resnet20(v2)	use	2	0.2	400	0.244	0.956	0.367	0.912	  
resnet20(v2)	use	2	0.3	200	0.261	0.950	0.371	0.911	  
resnet20(v2)	use	2	0.5	200	0.293	0.942	0.386	0.907	  
resnet20(v2)	use	2	1	200	0.361	0.928	0.433	0.900	  
resnet20(v2)	use	3	0.1	200	0.242	0.957	0.372	0.914	  
resnet20(v2)	use	3	0.1	400	0.244	0.954	0.370	0.911	  
resnet20(v2)	use	3	0.2	200	0.276	0.944	0.378	0.908  
