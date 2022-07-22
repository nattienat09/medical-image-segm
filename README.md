# Medical Image Segmentation
This is the code for my master's thesis. I conducted all my experiments within the Artificial Intelligence and Learning Systems Laboratory of the National Technical University of Athens in 2022.


Medical image segmentation involves identifying regions of interest in medical images. In modern times, there is a great need to develop robust computer vision algorithms to perform this task in order to reduce the time and cost of diagnosis and thus to aid quicker prevention and treatment of a variety of diseases. The approaches presented so far, mainly follow the U-type architecture proposed along with the UNet model, implement encoder-decoder type architectures with fully convolutional networks,and also transformer architectures, exploiting both attention mechanisms and residual learning, and emphasizing information gathering at different resolution scales. Many of these architectural variants achieve significant improvements in quantitative and qualitative results in comparison to the pioneer UNet, while some fail to outperform it. In this work, 11 models designed for medical image segmentation, and other types of segmentation, are trained and tested, evaluated on specific evaluation metrics, on four publicly available datasets related to gastric polyps and cell nuclei, which are first augmented to increase their size in an attempt to address the problem of the lack of a large amount of medical data. In addition, their generalisability and the effect of data augmentation on the scores of the experiments are also examined. Finally, conclusions on the performance of the models are provided and future extensions that can improve their performance in the task of medical image segmentation are discussed.

<p align="center">
  <img width="460" height="500" src="https://user-images.githubusercontent.com/48295759/180458910-10913506-fb1f-48de-a319-af9ce12a28a0.png">
</p>


All the code is implemented in Tensorflow 2.4 and Keras.

Each folder contains code for the model with the same title.

The "xnet" folder contains code for UNet, VNet, ResUNet-a, TransUNet, SwinUNet, R2UNet and Attention UNet.

The "data preproc" folder contains code for splitting the datasets in train, validation and test sets, for merging the masks of the 2018 Data Science Bowl dataset, for merging the masks of the SegPC dataset and for the data augmentation.
