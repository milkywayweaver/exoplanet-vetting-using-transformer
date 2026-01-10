# exoplanet-vetting-using-transformer
Exoplanet vetting model using multi input transformer approach.

### Introduction
Exoplanets are planet that lie outside of the Solar System and orbit stars other than the Sun. Ever since its discorvery in 1992, over 4000 exoplanets have been discovered through various means, such as trasnit method, radial velocity method, direct imaging, etc. Transit method has been the most productive methods at detecting an exoplanet. As of January 10th 2026, out of around 6000 confirmed exoplanets, 73.8% of the them are found using the transit method.

When an exoplanet crosses the stellar disk, some of the star light is blocked by the planet. We can detect an exoplanet by looking at the decrease in the star light observed, which can be done by continously observing the star and constructing a light curve. A light curve is a graph that represents light observed as a function of time.

But, not every decrease (or dip) in the light curve corresponds to an exoplanet. Other astrophysical phenomena can also cause dips in the light curve, such as eclipsing binaries or variable stars. Instrumental noise can also cause dips in the light curve. Researchers used to manually inspect every light curve and classify/vet it as either planet candidate or false positives. This method is prone to inconsistency between vetters (Yu, 2019). In more recent years, machine learning methods have been applied to aid with this problem. Some of the earlier attempts are Robovetter which uses decision tree and Autovetter which utilizes random forest algorithm. Later, Shallue & Vanderburg (2018) creates AstroNet which uses convolutional neural networks (CNN) to classify planet candidate and false positives, which achieves remarkable accuracy of around 96%. Ansdell et al (2018) adds domain knowledge information to AstroNet and called it ExoNet, which increases the accuracy by about 2%.

Ever since its introduction in 2017, transformers have reshaped a lot of state-of-the-art architectures in many fields, including natural language processing and computer vision. It utilizes attention to see the sequential influece of one data point to the others, making it suitable for sequential data. In this study, we tried to use transformer architecture to replace the CNN feature extractor of ExoNet.

### Dataset
We use the dataset used for ExoNet provided by Ansdell et all (2018), which includes Kepler light curves and centroid curves for both global and local views, as well as some stellar parameters such as stellar effective temperature, surface gravity, metallicity, etc. Then, we load the data into a single csv file that contains all the features, one file for one set (train, validation, and test). We do not perform further preprocessing as most has been done by Ansdell et al. Then we augment the data by randomly flipping the x-axis of the light curve and centroid curve for both the global and local views. The train set contains 11937 entries whereas the validation and test contains 1574 entries. In each set, only around 22% are labeled planet candidate and the rest are labeled false positives. 

### Reference
Ansdell, M., Ioannou, Y., Osborn, H. P., Sasdelli, M., Smith, J. C., Caldwell, D., ... & Angerhausen, D. (2018). Scientific domain knowledge improves exoplanet transit classification with deep learning. The Astrophysical journal letters, 869(1), L7.

Huang, S., & Jiang, C. (2025). Machine Learning for Exoplanet Discovery: Validating TESS Candidates and Identifying Planets in the Habitable Zone. arXiv preprint arXiv:2512.00967.

Malik, A., Moster, B. P., & Obermeier, C. (2022). Exoplanet detection using machine learning. Monthly Notices of the Royal Astronomical Society, 513(4), 5505-5516.

NASA Science. (n.d.). Exoplanet discoveries dashboard. Retrieved [January 10, 2026], from https://science.nasa.gov/exoplanets/discoveries-dashboard/

Shallue, C. J., & Vanderburg, A. (2018). Identifying exoplanets with deep learning: A five-planet resonant chain around kepler-80 and an eighth planet around kepler-90. The Astronomical Journal, 155(2), 94.

Yu, L., Vanderburg, A., Huang, C., Shallue, C. J., Crossfield, I. J., Gaudi, B. S., ... & Quinn, S. N. (2019). Identifying exoplanets with deep learning. III. Automated triage and vetting of TESS candidates. The Astronomical Journal, 158(1), 25.
