# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

[//]: # (Image References)
[image1]: ./writeup_images/histo_dataset.png "Visualization"
[image2]: ./writeup_images/image_samples.png "Visualization"
[image3]: ./writeup_images/prepro_original.png "Preprocessing"
[image4]: ./writeup_images/prepro_gray.png "Preprocessing"
[image5]: ./writeup_images/prepro_eqHist.png "Preprocessing"
[image6]: ./writeup_images/augmentation.png "Augmentation"
[image7]: ./writeup_images/failure_trend.png "Failure trend"
[image8]: ./writeup_images/accuracy_before.png "Accuracy"
[image9]: ./writeup_images/accuracy_final.png "Accuracy"
[image10]: ./new_test_data_from_web/2.jpg "Traffic Sign 1"
[image11]: ./new_test_data_from_web/23.jpg "Traffic Sign 2"
[image12]: ./new_test_data_from_web/28.jpg "Traffic Sign 3"
[image13]: ./new_test_data_from_web/30.jpg "Traffic Sign 4"
[image14]: ./new_test_data_from_web/33.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/shinsteve/udacity-carnd-trafficsign-p2/blob/master/Traffic_Sign_Classifier.ipynb)

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

![alt text][image3] ![alt text][image4]

Secondly, I applied "Equalize Histo" filter of OpenCV to impromve the contrast.

![alt text][image4] ![alt text][image5]

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

* Every class has the same number of data. (= max(train_num_per_class) * INCREASE_FACTOR)
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
| EPOCHS    | 10    |
| BATCH_SIZE | 128  |
| Learning rate | 0.001 |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My first architecture was almost the same as LeNet. But the validation accuracy was 0.920, which didn't meet the criteria.
I investigated the trend of failure case with respect to class type. As shown in the below graph, it seems that the number of training set for the class is small in case the failure rate of that class is high.

![alt text][image7]

To improve this situation, I adjusted the balance of data distribution wrt class type by data augmentation technique as described in the previous section.

Secondly I investigated how accuracy moves as epoch proceeds. As shown in the below graph, it seemed that so called overfitting, a high accuracy on the training set but low accuracy on the validation set, occurred.

![alt text][image8]

To improve this situation, I changed the architecture as follows:
* Add Dropout layer after the activation layers
* Taking away one fully connected layer

Finally, I adjusted the hyper parameters.
The value of hyper parameters are shown below.

| Parameter | Value |
|:---------:|:-----:|
| EPOCHS    | 14    |
| BATCH_SIZE | 128  |
| Learning rate | 0.001 |
| INCREASE_FACTOR | 1.5 |
| KEEP_PROB | 0.67 |

The final results were:
* training set accuracy of 97.0%
* validation set accuracy of 96.8%.
* test set accuracy of 93.5%

![alt text][image9]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]

The 4th image might be difficult to classify because part of the sign board is covered with snow and the figure is hidden.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)		| Speed limit (50km/h)   			| 
| Slippery road     			| Slippery road 							|
| Children crossing		| Children crossing							|
| Beware of ice/snow      		| Priority road	 				|
| Turn right ahead			| Turn right ahead  							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This doesn't compare favorably to the accuracy on the test set of 93.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the 1st image, the model is relatively sure that this is a "Speed limit (50km/h)" sign (probability of 0.737),and the image does contain a Speed limit (50km/h) sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       				|
|:---------------------:|:---------------------------------------------:|
| .74         			| Speed limit (50km/h)   		|
| .17         			| Speed limit (80km/h)   		|
| .5         			| Speed limit (30km/h)   		|
| .3         			| Speed limit (60km/h)   		|
| .0         			| Speed limit (100km/h)   		|

For the 2nd image, the model is relatively sure that this is a "Slippery road" sign (probability of 1.000),and the image does contain a Slippery road sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       				|
|:---------------------:|:---------------------------------------------:|
| .100         			| Slippery road   		|
| .0         			| Dangerous curve to the right   		|
| .0         			| Dangerous curve to the left   		|
| .0         			| Vehicles over 3.5 metric tons prohibited   		|
| .0         			| No passing for vehicles over 3.5 metric tons   		|

For the 3rd image, the model is relatively sure that this is a "Children crossing" sign (probability of 0.836),and the image does contain a Children crossing sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       				|
|:---------------------:|:---------------------------------------------:|
| .84         			| Children crossing   		|
| .9         			| Bicycles crossing   		|
| .3         			| Dangerous curve to the right   		|
| .2         			| Ahead only   		|
| .1         			| Beware of ice/snow   		|

For the 4th image, the model is relatively sure that this is a "Priority road" sign (probability of 0.903),but the image contain a Beware of ice/snow sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       				|
|:---------------------:|:---------------------------------------------:|
| .90         			| Priority road   		|
| .6         			| Roundabout mandatory   		|
| .2         			| Right-of-way at the next intersection   		|
| .1         			| End of no passing by vehicles over 3.5 metric tons   		|
| .0         			| Speed limit (60km/h)   		|

For the 5th image, the model is relatively sure that this is a "Turn right ahead" sign (probability of 0.999),and the image does contain a Turn right ahead sign. The top five soft max probabilities were:

| Probability         	|     Prediction	       				|
|:---------------------:|:---------------------------------------------:|
| .100         			| Turn right ahead   		|
| .0         			| Ahead only   		|
| .0         			| Speed limit (60km/h)   		|
| .0         			| Keep left   		|
| .0         			| Road work   		| 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
(TODO)This is future work at this moment.


