# Convolutional Neural Networks for Task-evoked fMRI Data Classification
M2DCNN is a repository of codes and experiment results for A Multi-channel 2D Convolutional Neural Networks Model for Task-evoked fMRI Data Classification, using Keras (version 2.2.4) with TensorFlow (version 1.12.0) as backend.
### Publications
See the following publications for examples of this code in use:
 * **A Multi-channel 2D Convolutional Neural Networks Model for Task-evoked fMRI Data Classification.** Jinlong Hu, Yuezhen Kuang, Bin Liao, Lijie Cao, Shoubin Dong, Ping Li, [Computational Intelligence and Neuroscience](https://new.hindawi.com/journals/cin/2019/5065214/), 2019.

### Codes

[M2D_CNN_model.py](M2D_CNN_model.py) is the Python code of M2D CNN model.  
[cnn3d_model.py](cnn3d_model.py) is the Python code of 3D CNN model.  
[sep3d_model.py](sep3d_model.py) is the Python code of 3D SepConv model. *To run this model, you should import SeparableConv3D from [sepconv3D](https://github.com/simeon-spasov/MCI/tree/master/utils/sepConv3D.py).*  
[s2D_CNN_model.py](s2D_CNN_model.py) is the Python code of s2D CNN model.  
[mv2D_CNN_model.py](mv2D_CNN_model.py) is the Python code of mv2D CNN model.  
[cnn1d_model.py](cnn1d_model.py) is the Python code of 1D CNN model.  
[svm_model.py](svm_model.py) is the Python code of SVM model.  

### Experiment results
 
#### Classification performance
9950 samples from 995 subjects (mean±std):  

Model	| Accuracy | Precision | F1-Score
------ | ------- | -------- | ------
PCA+SVM	| 48.94±2.36%	| 48.17±2.48%	| 0.4779±0.0232
mv2D CNN	| 63.36±2.19%	| 63.59±2.27%	| 0.6306±0.0222
3D CNN	| 82.34±1.27%	| 82.68±1.39%	| 0.8239±0.0130
3D SepConv	| 80.44±1.16%	| 80.88±1.24%	| 0.8043±0.0116
1D CNN	| 80.76±1.69%	| 80.94±1.73%	| 0.8068±0.0178
s2D CNN	| 81.80±0.89%	| 81.95±0.97%	| 0.8179±0.0094
M2D CNN	| 83.20±2.29%	| 83.63±1.87%	| 0.8321±0.0223



#### Training time
2000 samples from 200 subjects (mean±std):  

Model |	Training time (S) |	Total number of epochs
------ | ------- | -------- 
mv2D CNN | 909±134 | 54±8
3D CNN | 1156±185 | 39±6
3D SepConv | 1601±196 | 41±5
1D CNN | 834±157 | 39±7
s2D CNN | 565±102 | 31±6
M2D CNN | 1074±348 | 39±13

#### training and validation losses
2000 samples from 200 subjects:    
![loss-2000](200-Loss-mean-std-plot.png)  
5000 samples from 500 subjects：  
![loss-5000](500-Loss-mean-std-plot.png)  
