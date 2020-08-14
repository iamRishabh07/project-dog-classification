# Dog-Breed Classifier
The dog-breed classifier project is a part of Udacity Deep Learning Nanodegree.

This project uses Convolutional Neural Networks (CNN) to identify the canine’s breed.

<div align="center">
<img src="https://s3.amazonaws.com/video.udacity-data.com/topher/2018/April/5adec8b9_cnn-project/cnn-project.jpg" height="300" width="300" />
<br />
<h1>Dog-Breed Classifier</h1>
</div>

## Project Overview

In this project, I have learned how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, the algorithm will identify an estimate of the canine’s breed. If supplied an image of a human face, the code will identify the resembling dog breed.

Along with exploring state-of-the-art CNN models for classification, I have made important design decisions about the user experience for the app. By completing this project, I have demonstrated my understanding of the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.


## Step 0: Import Datasets

## Step 1: Detect Humans

## Step 2: Detect Dogs

## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
I started by taking a look at already existing architectures to see what works.

#### CIFAR
* Input: 32x32 RGB
* Output: 10 classes

```
Net(
  (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=1024, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=10, bias=True)
  (dropout): Dropout(p=0.25)

```

#### VGG-16
I also looked at VGG-16 and VGG-19 architectures, but training them would probably take way too long.

### Best practices
Then, I researched existing best practices.   

I read these articles on best practices:
* https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
* https://www.topbots.com/14-design-patterns-improve-convolutional-neural-network-cnn-architecture/


and followed these recommendations:
* "Always start by using smaller filters is to collect as much local information as possible"
* "...the number of channels generally increase or stay the same while we progress through layers in our convolutional neural net architecture"
* "Always use classic networks like LeNet, AlexNet, VGG-16, VGG-19 etc. as an inspiration while building the architectures for your models. "
  * I used VGG-16 network as inspiration

#### Our model
Input and output:
* Input: 224x224 RGB
* Output: 133 classes

Finally, based on the architecture I've already seen and the best practices, I came up to this architecure.  
Of course, it took several steps (you can find details below), so finally I chose the simplest architecture that showed good results.
```
Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=6272, out_features=500, bias=True)
  (fc2): Linear(in_features=500, out_features=133, bias=True)
  (dropout): Dropout(p=0.25)
)
```

## Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
> Outline the steps you took to get to your final CNN architecture and your reasoning at each step.

The dogs dataset uses 133 classes, and VGG16 is capable of identifying 118 dog breeds.  
Therefore, I started by using VGG16 and change its classifiers to recognize 133 dog breeds (133 is just a bit more than 118). Unfortunately, it didn't give good accuracy on test set.  
I tried VGG-19, it didn't perform well on the test set either (only 57%).

Finally, I looked at the information about errors for the topc ImageNet architectures provided by this table: https://pytorch.org/docs/stable/torchvision/models.html   
  
When I ran the new network for the first time, I started with a terrible validation loss (>4) - which was not expected at all as it's already pretrained. Probably, it was this intermittent problem with strange results as I described above. 
So, I stopped execution and started again, I also changed optimizer from SGD to Adam.  
It started with a promising validation loss (2.823208) and showed good results on the test set:  
* Test Loss: 0.964425
* Test Accuracy: 80% (666/836)

> Describe why you think the architecture is suitable for the current problem.  

ResNet shows excellent results on ImageNet, so it's a natural choice. I tried to get the simplest architecture possiblem, so started by changing only the last classifier layer. It worked.

## Step 5: Write your Algorithm

## Step 6: Test Your Algorithm


# Chalenges
## Issues with reproducibility
I started with using the Jupyther notebook supplied as a part of the lesson. But! Running the same code was producing completely different results. I tried restarting the kernel,but it didn't help.  
To have more flexibilty, I switched to a notebook ran on AWS where I could also restart the notebook and even reboot the machine. Still, running the same code was producing diffferent results. One time it would get stuck at validation loss around 4.8 for thr first 10 epochs (and then I would shut it down).   
Another time, using the same code, it would learn very efficiently and obtain validation loss below 4.0 by 10th epoch. 

## Problems with CUDA on AWS AMIs

Udacity's lesson about [Cloud Computing](Deep Learning AMI with Source Code) recommends using the "Deep Learning AMI with Source Code (CUDA 8, Ubuntu)" AMI run notebooks.   
CUDA 8 is old, and AMI for CUDA 9 is available. 

#### Using CUDA 8 AMI 
`torch.cuda.is_available()` returns false, and IIRC there are other conflicts with torch and torchvision.

#### Using CUDA 9 AMI 
As I had problems with using CUDA on the older AMI, I installed a newer one. 
Initially, `torch.cuda.is_available()` was returning true (as expected), but torch wasn't installed even though I initiated the dependencies from requirements.txt.  

After installing torch, `torch.cuda.is_available()` became false and computation became __very__ slow.   
The problem is described here: described here: https://github.com/pytorch/pytorch/issues/15612   
I followed their advice and installed a nightly build of torch which contained the fix: 
```
!pip3 install torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
```
It worked, but cuda got disabled after the following EC2 instance restart. After the restart, even using the nightly buyild wasn't helping any more. So, I updated the driver as described here:  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html

## Authors

* Rishabh Srivastava

## License

[MIT](https://choosealicense.com/licenses/mit/)

