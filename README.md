# Exoplanet Vetter Using Transformers

### Introduction
Exoplanets are planet that lie outside of the Solar System and orbit stars other than the Sun. Ever since its discorvery in 1992, over 4000 exoplanets have been discovered through various means, such as transit method, radial velocity method, direct imaging, etc. Transit method has been the most productive methods at detecting an exoplanet. As of January 10th 2026, out of around 6000 confirmed exoplanets, 73.8% of the them are found using the transit method.

When an exoplanet crosses the stellar disk, some of the star light is blocked by the planet. We can detect an exoplanet by looking at the decrease in the star light observed, which can be done by continously observing the star and constructing a light curve. A light curve is a graph that represents light observed as a function of time.

But, not every decrease (or dip) in the light curve corresponds to an exoplanet. Other astrophysical phenomena can also cause dips in the light curve, such as eclipsing binaries or variable stars. Instrumental noise can also cause dips in the light curve. Researchers used to manually inspect every light curve and classify/vet it as either planet candidate or false positives. This method is prone to inconsistency between vetters (Yu, 2019). In more recent years, machine learning methods have been applied to aid with this problem. Some of the earlier attempts are Robovetter which uses decision tree and Autovetter which utilizes random forest algorithm. Later, Shallue & Vanderburg (2018) creates AstroNet which uses convolutional neural networks (CNN) to classify planet candidate and false positives, which achieves remarkable accuracy of around 96%. Ansdell et al (2018) adds domain knowledge information to AstroNet and called it ExoNet, which increases the accuracy by about 2%.
<br>
<img width="800" height="341" alt="image" src="https://github.com/user-attachments/assets/4e668356-7fa3-49be-bc33-ada110111772" />
<br>
_Figure 1. Exoplanet transit light curve visualization (NASA/Ames Research Center)_
<br>

Ever since its introduction in 2017, transformers have reshaped a lot of state-of-the-art architectures in many fields, including natural language processing and computer vision. It utilizes attention to see the sequential influece of one data point to the others, making it suitable for sequential data. In this study, we tried to use transformer architecture to replace the CNN feature extractor of ExoNet.

### Dataset
We use the dataset used for ExoNet provided by Ansdell et all (2018), which includes Kepler light curves and centroid curves for both global and local views, as well as some stellar parameters such as stellar effective temperature, surface gravity, metallicity, etc. Then, we load the data into a single csv file that contains all the features, one file for one set (train, validation, and test). We do not perform further preprocessing as most has been done by Ansdell et al. Then we augment the data by randomly flipping the x-axis of the light curve and centroid curve for both the global and local views. Unlike the work of Ansdell, we do not give random gaussian noise to the dataset. The train set contains 11937 entries whereas the validation and test contains 1574 entries. In each set, only around 22% are labeled planet candidate and the rest are labeled false positives.

### Architecure
The overall structure of the model is similar to ExoNet, except that we replace the CNN feature extractor with transformer encoders. To create the embeddings for the global and local views, we pass the data into a single convolutional layer with kernel size of 5, padding of 2, and stride of 1, before passing it through an average pooling layer. The result is an embedding of the features in E dimensional space with length L. Then trainable positional encoding is added to preserve order. The transformer encoder is inspired by the vision transformer (Dosovitskiy, 2020)) implementation of the encoder, which applies the layer normalization before the multihead attention (MHA) and multi layer perceptron (MLP). Residual connection is then added after each MHA and MLP. After the all of the encoder blocks, the CLS_token is taken from each view, and then concatenated with the stellar parameters before passing it through another MLP head. Sigmoid activation function is used for the output activation function. 

<img width="3508" height="2481" alt="Exoplanet Transformer Schematic" src="https://github.com/user-attachments/assets/443d09d1-9fcb-4686-bd75-b004baa06ee3" />

_Figure 2. Architecture of the model used in this study._

### Results
We train the model with the following hyperparameters:
* Batch Size: 32
* Embedding Dimension: 64
* Number of Heads: 8
* Dropout Rate: 0.5
* MLP Hidden Size: 256
* Number of Encoders: 4
* Global Embed Length: 512
* Local Embed Length: 256
* Number of Classes: 1
* Learning Rate: 1e-05
* Epochs: 100 <br>

We got accuracy of 0.9638, recall of 0.9417, and precision of 0.9040. The accuracy is around 1% better than AstroNet at 0.958 (Shallue & Vanderburg, 2018) but still 1% lower than ExoNet at 0.975 (Ansdell et al, 2018). Note that we give different augmentation to the dataset, so this comparison is not 100% apple to apple. Below is the confusion matrix on the test set using this model.
<br>
<img width="700" height="600" alt="8_2_confmat" src="https://github.com/user-attachments/assets/5c064f72-ed72-46fa-ac17-aec39e2a8069" />
<br>
_Figure 3. Confusion matrix of transformer model on the test set. x-axis represents the predicted label and the y-axis represents the true label._
<br>
<br>
We also tried ExoNet provided by Ansdell et al using our dataset format and augmentation. We modified the code a little to fit the format of the dataset we use. In our testing, ExoNet still performs better than our transformer model, with accuracy of 0.9771, recall of 0.9583, and precision of 0.9426. Below is the confusion matrix of the ExoNet model.
<br>
<img width="700" height="600" alt="exonet_confmat" src="https://github.com/user-attachments/assets/a1581c5f-4e2b-47c4-9835-0671604ed5ee" />
<br>
_Figure 4. Confusion matrix of ExoNet model on the test set. x-axis represents the predicted label and the y-axis represents the true label._
<br>

### Conclusion
We tried applying transformer architecture as feature extractor as opposed to CNN in ExoNet model. Though it works and achieves relatively good accuracy, it could not surpass ExoNet just yet. There are numbers of ways to improve the transformer model, such as examining different hyperparameters, changing the transformer implementation, etc.

### Credits & Resources
This project was developed following the educational framework provided by:
* **Instructor:** Mohammed Fahd Abrah
* **Platform:** [freeCodeCamp.org](https://www.youtube.com/c/Freecodecamp)
* **Course:** [Building a Vision Transformer Model from Scratch with PyTorch](https://youtu.be/7o1jpvapaT0)

For the dataset, please refer to the original ExoNet paper by Ansdell et al (2018), where they provided the link to the dataset.

### Reference
Ansdell, M., Ioannou, Y., Osborn, H. P., Sasdelli, M., Smith, J. C., Caldwell, D., ... & Angerhausen, D. (2018). Scientific domain knowledge improves exoplanet transit classification with deep learning. The Astrophysical journal letters, 869(1), L7.

Dosovitskiy, A. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

freeCodeCamp.org. (2025, May 7). Building a Vision Transformer Model from Scratch with PyTorch [Video]. YouTube. https://youtu.be/7o1jpvapaT0?si=ATyNV0c_Cyd82y-U

Huang, S., & Jiang, C. (2025). Machine Learning for Exoplanet Discovery: Validating TESS Candidates and Identifying Planets in the Habitable Zone. arXiv preprint arXiv:2512.00967.

Malik, A., Moster, B. P., & Obermeier, C. (2022). Exoplanet detection using machine learning. Monthly Notices of the Royal Astronomical Society, 513(4), 5505-5516.

NASA Science. (n.d.). Exoplanet discoveries dashboard. Retrieved [January 10, 2026], from https://science.nasa.gov/exoplanets/discoveries-dashboard/

Shallue, C. J., & Vanderburg, A. (2018). Identifying exoplanets with deep learning: A five-planet resonant chain around kepler-80 and an eighth planet around kepler-90. The Astronomical Journal, 155(2), 94.

Yu, L., Vanderburg, A., Huang, C., Shallue, C. J., Crossfield, I. J., Gaudi, B. S., ... & Quinn, S. N. (2019). Identifying exoplanets with deep learning. III. Automated triage and vetting of TESS candidates. The Astronomical Journal, 158(1), 25.
