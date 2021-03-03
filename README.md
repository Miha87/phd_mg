#Phd repo
Repo contains material created during experiments that were part of the Ph.D. thesis.

The research was motivated by weaknesses of traditional approaches to time study,     
which are time-consuming and highly subjective. Mentioned motivation reflects my   
background in industrial engineering.  

The goal of this research was the development of a deep learning model with the capability    
of recognition and temporal segmentation of a series of human activities from videos collected     
in manufacturing processes. This problem is usually called **"action segmentation"** or     
**"action detection"**.

Model inputs were videos of the maximum duration of up to 2 minutes.

Examples of model output:  
![25_gif](https://user-images.githubusercontent.com/34508474/109804982-2fa56480-7c23-11eb-86a3-8c17f60f4261.gif)
![196_gif](https://user-images.githubusercontent.com/34508474/109804991-3338eb80-7c23-11eb-9cb2-cb6c99a60b1d.gif)  
![256_gif](https://user-images.githubusercontent.com/34508474/109805003-3633dc00-7c23-11eb-9815-57abe2f80911.gif)
![327_gif](https://user-images.githubusercontent.com/34508474/109805011-37fd9f80-7c23-11eb-8188-ab54e32b81dc.gif)

Additional output:  
![image](https://user-images.githubusercontent.com/34508474/109805975-662faf00-7c24-11eb-8d07-5139e87bfb6d.png)

To achieve this goal, a sample was collected from the real manufacturing process,  
which consists of nine work activities. Approximately 40 hours of video recording were 
collected. During the video recording of the process, the work activities were performed   
by four subjects on three different types of products, while the recording itself was 
performed from two different view positions. 27 different models have been developed which differ 
with respect to recording viewpoint, model input features, and model architecture responsible 
for activity classification and time segmentation.  

The main parts of the repository are:
* data_prep - data preparation process (ipynb)
* stat_analiza - statistical analysis of the collected sample (R)
* phd_research - it contains a developed library **phd_lib** based on TF 2   
for easy application of deep learning to video data and scripts in 
which phd_lib was applied during experiments (Py)




