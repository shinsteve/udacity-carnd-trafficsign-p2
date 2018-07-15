# **Traffic Sign Recognition** 

---

## MEMO
- validation accuracy
  - Initial LeNet : 0.872
  - Grayscale : 0.872
  - 0-1 normalize : 0.887
  - -0.5-0.5 normalize : 0.892
  - norm + equalizeHist : 0.926
  - norm + 0 mean scale : 0.926
  - norm + equalizeHist + 0 mean scale : 0.931

`
EPOCHS = 13
BATCH_SIZE = 128
rate = 0.001
KEEP_PROB = 0.67
Accuracy: Train = 0.972,   Validation = 0.970
Test Accuracy = 0.936
`

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

TODO update image

[//]: # (Image References)
[image1]: ./writeup_images/histo_dataset.jpg "Visualization"
[image2]: ./writeup_images/image_samples.jpg "Visualization"
[image3]: ./writeup_images/prepro_original.jpg "Preprocessing"
[image4]: ./writeup_images/prepro_gray.jpg "Preprocessing"
[image5]: ./writeup_images/prepro_eqHist.jpg "Preprocessing"
[image6]: ./writeup_images/augmentation.jpg "Augmentation"
[image10]: ./new_test_data_from_web/.png "Traffic Sign 1"
[image11]: ./new_test_data_from_web/.png "Traffic Sign 2"
[image12]: ./new_test_data_from_web/.png "Traffic Sign 3"
[image13]: ./new_test_data_from_web/.png "Traffic Sign 4"
[image14]: ./new_test_data_from_web/.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shinsteve/udacity-carnd-trafficsign-p2/blob/master/Traffic_Sign_Classifier.ipynb)

(TODO) merge branch

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python/numpy to calculate summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
The first one is a bar chart showing how the data are distributed wrt class type. As you can see, there exists significant bias and distribution is not uniform.

![alt text][image1]

The 2nd one is showing sample images for each class. As you can see, some of the images has bad contrast and it is not easy to judge the class of it even by human.

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

##### Pre-processing image

As a first step, I decided to convert the images to grayscale because it seems shape rather than color matters in classifying traffic sign.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]

Secondly, I applied "Equalize Histo" filter of OpenCV to impromve the contrast.

![alt text][image4]
![alt text][image5]

As a last step, I normalized the image data because it's suited to the activation function and learning algorithm of the neural network. Normalizaton consists of the followin 2 portion.

1. Convert to floating point ranging from -0.5 to 0.5 (0 center)
1. Linearly scales image to have zero mean and unit norm. (tf.image.per_image_standardization)


##### Data augmentation

I decided to generate additional data because data distributed wrt class type in the train set is not uniform.

To add more data to the the data set, I used the following techniques because they generated a little different images than the original image, and that I think prevents the model from overfitting.

* Crop and zoom randomly upto 20%.
* Rotate randomly in 20 degree range.
* Random noise is added.

Here is an example of an original image and an augmented image:

![alt text][image6]

The augmented data set is different from original data set in the following points:

* Every class has the same number of data. (= max(train_num_per_class) * INCREASE_FACTOR
* The increased images are generated by the aumentaion techniques above.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Dropout				| Keep_prob = 0.67								|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  				|
| Dropout				| Keep_prob = 0.67								|
| Flattern				| outputs 400									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| Dropout				| Keep_prob = 0.67								|
| Fully connected		| outputs 43 									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an "Adam" optimizer to minimize the loss that is defined as the mean of cross entropy of output logits.

The value of hyper parameters are shown below.

| Parameter | Value |
|:---------:|:-----:|
| EPOCHS    | 13    |
| BATCH_SIZE | 128  |
| Learning rate | 0.001 |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

overfitting -> dropout
overfitting -> remove fully connected layer
Parameter tuning


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
(TODO) update the sign
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the (TODO) cell of the Ipython notebook.

TODO: code to generate result string

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
(TODO)This is future work at this moment.


