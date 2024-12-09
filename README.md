# Mamba_VLM
Part one is extension of Mamba to medical image classification. Part two is to optimize VLM for improved efficiency.


### Mamba_extension.py
Mamba is combined together with CNN backbone to complete the task of classifying medical images in MHIST dataset. The location of dataset should be placed in the following arrangement: "./mhist_dataset/mhist_images" it the folder to all MHIST images and "./mhist_dataset/annotations.csv" is the path to corresponding csv file. 

### CNN.py
This is the baseline in which only CNN is implemented to classify the MHIST data. 


### project_2.py
The second part of project is completed here where the FLOPs of optimized VLM is compared with the baseline VLM.


