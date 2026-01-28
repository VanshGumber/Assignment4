# Assignment4 - Learning Probability Density Functions using data only
1. Methodology

Data Loading → Preprocessing → Roll-Number-Based Transformation →
GAN Training → Sample Generation → PDF Estimation → Analysis

2. Dataset Information

Dataset: India Air Quality Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data
Feature Used: NO₂ 

3. Objective
To learn an unknown probability density function of a transformed random variable using a
Generative Adversarial Network (GAN).

4. Mathematical Formulation
Each NO₂ value (x) is transformed as:
z = x + a * sin(bx)
where:
a = 0.5 * (r mod 7)
b = 0.3 * (r mod 5 + 1)
and r = roll number (102303922)

Transformation Parameters
For the given roll number:
a = 0
b = 0.9

Since a = 0 the transformation reduces down to z = x

5. GAN Architecture
Generator: Fully connected network mapping 1D Gaussian noise to samples of 𝑧
Discriminator: Fully connected network classifying real vs generated samples
Both networks are trained adversarially

6. PDF Approximation
After training:
A large number of samples are generated from the generator
Kernel Density Estimation (KDE) is applied
The resulting curve represents the estimated PDF p(z)

7. Results and Observations
Mode Coverage:
The generator captures the dominant mode of the distribution.
Training Stability:
Training remains stable due to normalization and the low-dimensional GAN design.
Quality of Generated Distribution:
The estimated density closely follows the empirical distribution, with minor deviations in the tails.

8. Conclusion
This assignment demonstrates that GANs can learn an unknown probability density function directly from data samples without assuming any analytical form, providing a data-driven solution for density estimation.

9. Tools & Platform
Python, NumPy, Pandas
PyTorch
Matplotlib, Seaborn
Google Colab
