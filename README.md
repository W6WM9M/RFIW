# CX4041 Machine Learning Project: Recognizing Families In the Wild
<p align="justify"> 
This Github repository contains the work done for the CX4041 Machine Learning project in Nanyang Technological University, Singapore. In this project, we are required to work on one of the past Kaggle competitions by applying what we have learnt during the course of the semester. We decided to work on the 2019 Kaggle Competition: <a href="https://www.kaggle.com/competitions/recognizing-faces-in-the-wild/overview">Northeastern SMILE Lab - Recognising Faces in the Wild</a>, where we are provided with the <a href="https://web.northeastern.edu/smilelab/fiw/">Families In the Wild</a> dataset and tasked to tackle the problem of Facial Kinship Verification (FKV). The goal of FKV is to create an automatic kinship classifier to determine if two individuals are biologically related based on just their facial images. Submissions to Kaggle were evaluated on the area under the ROC curve between the predicted probability and the observed target.
</p>

<p align="justify"> 
After extensive research on the performance of traditional machine learning techniques and deep learning techniques, we decided to concentrate our efforts on transfer learning, which is well-known to generate a good performance within a relatively short period of time. The use of transfer learning in our project revolves around utilizing models previously trained for facial recognition tasks to be customized to fit our FKV needs. Moreover, we also used count on ensemble learning to boost our test results. 
</p>

<p align="justify"> 
In our experiments, we attempted various pretrained state-of-the-art face models, which include <a href="https://ieeexplore.ieee.org/document/6909616">Facebook DeepFace</a>, <a href="https://ieeexplore.ieee.org/document/8953658">ArcFace</a>, <a href="https://ieeexplore.ieee.org/document/7298682">FaceNet</a>, <a href="https://ieeexplore.ieee.org/document/7477553">OpenFace</a>, <a href="https://ieeexplore.ieee.org/document/7780459">ResNet</a>, <a href="https://ieeexplore.ieee.org/document/8578843">SENet</a>, and <a href="https://arxiv.org/abs/2103.14803">Face Transformer</a>. For each model, we tried one or more feature concatenation techniques and trained on different samples of the dataset to generate uncorrelated models to build our ensemble. Ultimately, throughout the duration of the project, we trained a total of 238 different base classifiers. Our best performing ensemble consists of a combination of classifiers trained on the outputs of ArcFace, FaceNet, ResNet-50, SENet-50, and Vision Transformer to produce a public score of 91.5%, placing us in the 6th place (out of 522 teams) in the public leaderboard. 
</p>

# Creating Uncorrelated Base Classifiers
To create uncorrelated base classifiers, we attempted the following methods:
<dl>
  <dt><b>1. Using Different Pretrained Feature Extractors</b></dt>
  <dd>Due to differences in training and architecture, different pretrained models extract different information about an image. As such, for each image pair, we can use different forms of feature vectors for classification.</dd>

  <dt><b>2. Using Multiple Feature Concatenation Methods</b></dt>
  <dd>We experimented with various methods of feature concatenations such as simple concatenation, absolute difference, squared absolute difference, exponential difference, and exponential ratio difference.</dd>
  
  <dt><b>3. Using K-Fold Cross Validation</b></dt>
  <dd>The training dataset was divided into K segments, where (K-1) segments are used to train our classifier while the remaining one segment is used for validation. This allowed us to generate K uncorrelated classifiers per feature concatenation method and per feature extractor.</dd>
</dl>

# Ensemble Methods
To combine the results obtained by the different base classifiers, we attempted the following ensemble methods: 
<dl>
  <dt><b>1. Averaging across Different Validation Sets</b></dt> 
  <dd>To combine the K classifiers genereated by K-fold cross validation, we simply average the classifiers' output probabilities on the test dataset.</dd> 
  
  <dt><b>2. Stacking to Combine Different Feature Concatenation Methods</b></dt>
  <dd>To combine multiple feature concatenation methods, we attempted stacking, where a combiner is trained to combine the different classifiers' outputs. In particular, we trained three different combiners including Sklearn Gradient Boosting, LightGBM, and a simple Neural Network.</dd> 
  
  <dt><b>3. Averaging across Different Feature Extractors</b></dt>
  <dd>To combine the ensembled result of each pretrained model, we attempted various combinations of averaging the ensemble results of the different models.</dd>
</dl> 

# Our Best Ensemble
The following shows how we obtained our highesst public score of 91.5:

