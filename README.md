# 229FinalProject
An investigation of ML and Generative AI techniques to diagnose AD severity

## Project Introduction 
Alzheimer's Disease (AD) is a neurodegenerative disease that may be 
categorized into 4 severity classes. This project explores multiple machine learning 
and deep learning approaches to classify MRI brain scans into different stages of 
Alzheimer's Disease severity, employing techniques ranging from traditional tree-based 
models to advanced neural networks and generative models. 


## Datasets
The image [dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset/data)
consists of MRI images of brains clinically diagnosed into one of four
categories of AD severity. The data has four classes:

1. Mild Demented
2. Moderate Demented
3. Non Demented
4. Very Mild Demented 


## Project Setup

1. Download the Dataset above and unzip the directory.

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
  - Conda is a package manager that sandboxes your project’s dependencies in a virtual environment
  - Miniconda contains Conda and its dependencies with no extra packages by default (as opposed to Anaconda, which installs some extra packages)
2. Extract the zip file and run `conda env create -f environment.yml` from inside the extracted directory.
  - This creates a Conda environment called `229Final`
3. Run `source activate 229Final`
  - This activates the `229Final` environment
  - Do this each time you want to write/test your code
4. (Optional) If you use PyCharm:
  - Open the `src` directory in PyCharm
  - Go to `PyCharm` > `Preferences` > `Project` > `Project interpreter`
  - Click the gear in the top-right corner, then `Add`
  - Select `Conda environment` > `Existing environment` > Button on the right with `…`
  - Select `/Users/YOUR_USERNAME/miniconda3/envs/Final229/bin/python`
  - Select `OK` then `Apply`
5. If modifications are made to the `environment.yml` file, run `conda env update --file environment.yml`
to update the conda environment.

## Files organization

The 'Augmentation' folder contains all the code, plots, and loss csv files for the simple augmentation task.
The 'XGB Classification' contains the features, training, testing, and the model for the multi-class classification using XGBoost.
The 'CNN' folder contains various CNN architectures and training notebooks with different optimization strategies.
The 'GAN' folder contains the Wasserstein GAN implementation for generating synthetic MRI images.
The 'Tree_Models' folder contains decision tree and XGBoost implementations.
The 'pre_processing' folder contains data preprocessing scripts and custom dataset classes.
The data was too heavy to be included here. The datasets are available on Kaggle and the code is given to obtain the augmented datasets and resized images.

## Machine Learning Techniques

This project implements and compares several state-of-the-art machine learning techniques for Alzheimer's Disease classification:

### 1. Data Augmentation
**Location:** `Augmentation/simple_augment.py`

To address class imbalance and improve model generalization, we implemented systematic data augmentation:
- **Techniques:** Random affine transformations (rotation ±15°, translation up to 10%, shear up to 10°)
- **Strategy:** Grid search over augmentation parameters (A, B, C, D) representing the number of augmented samples per class
- **Preprocessing:** Image resizing to 224×224, grayscale conversion with 3-channel output
- **Evaluation:** Tested multiple augmentation combinations to find optimal balance
- **Results:** Loss curves and model performance tracked across different augmentation strategies

### 2. Convolutional Neural Networks (CNNs)
**Location:** `CNN/` folder

Multiple CNN architectures were implemented and compared:

#### a) Transfer Learning with ResNet
- **ResNet18 & ResNet50:** Pre-trained models fine-tuned for 4-class AD classification
- **Modifications:** Final fully connected layer adapted for 4 output classes
- **Input:** 224×224 RGB images normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### b) Optimization Strategies
Multiple optimization approaches were explored:
- **Adam Optimizer:** Adaptive learning rate with lr=0.001
- **SGD with Momentum:** Classical optimization with momentum term
- **RMSProp:** Root Mean Square Propagation for adaptive learning rates
- **Learning Rate Decay:** StepLR scheduler (step_size=3, gamma=0.1) to gradually reduce learning rate
- **Extended Training:** Models trained for 5-20 epochs with different configurations

#### c) Attention Mechanisms
**Location:** `CNN/attention-resnet18-more_epochs_with_lr_decay_only.ipynb`
- Attention layers added to ResNet18 to focus on relevant brain regions
- Helps model identify diagnostically important features in MRI scans

#### d) Training Enhancements
- **Batch Size:** 32 (optimized for memory and convergence)
- **Loss Function:** Cross-Entropy Loss for multi-class classification
- **Early Stopping:** Monitored validation loss to prevent overfitting
- **Data Loading:** Multi-threaded data loading (num_workers=12) for efficiency

### 3. Generative Adversarial Networks (GANs)
**Location:** `GAN/2_gan_models/wgan_bs_32_lat_128_eps_600.py`

#### Wasserstein GAN with Gradient Penalty (WGAN-GP)
- **Purpose:** Generate synthetic MRI images to augment training data and address class imbalance
- **Architecture:**
  - **Generator:** Deep convolutional transpose network with batch normalization
    - Input: 128-dimensional latent vector
    - Output: 192×160 grayscale MRI image
    - Layers: Dense → Reshape → 5 Conv2DTranspose blocks with BatchNorm and LeakyReLU
    - Final activation: Tanh (output range [-1, 1])
  
  - **Discriminator (Critic):** Convolutional network without batch normalization
    - Input: 192×160×1 MRI image
    - Architecture: 4 convolutional blocks with LeakyReLU
    - Output: Single real-valued score (no sigmoid activation)
    - Initialization: He normal initialization for stable training

- **Training Strategy:**
  - **Loss Function:** Wasserstein loss with gradient penalty (λ=10)
  - **Training Ratio:** 5 discriminator updates per generator update
  - **Optimizer:** Adam (lr=0.0001, β₁=0.5, β₂=0.9)
  - **Batch Size:** 32
  - **Epochs:** 600-800
  - **Gradient Penalty:** Enforces 1-Lipschitz constraint for stable training

- **Key Features:**
  - Random weighted averaging between real and fake samples for gradient penalty
  - Separate model training for different AD severity classes (AD vs Normal)
  - Generated images normalized to [-1, 1] range
  - Periodic saving of generator weights and sample images

### 4. Tree-Based Models
**Location:** `Tree_Models/TreeModels.ipynb`, `XGB Classification/`

#### XGBoost Multi-Class Classification
- **Feature Extraction Methods:**
  
  1. **Deep Learning Features (Primary):**
     - Pre-trained ResNet50 used as feature extractor
     - Extract 1000-dimensional feature vectors from final layer
     - Features capture high-level semantic information from MRI scans
  
  2. **Texture Features:**
     - Haralick texture features computed using mahotas library
     - Gray-Level Co-occurrence Matrix (GLCM) based features
     - Captures local texture patterns in brain tissue

- **XGBoost Configuration:**
  - **Objective:** multi:softmax for 4-class classification
  - **Evaluation Metric:** Multi-class log loss (mlogloss)
  - **Hyperparameters:**
    - max_depth: 10
    - learning rate (eta): 0.2
    - num_boost_rounds: 100
  - **Regularization:** Early stopping with 10 rounds
  
- **Cross-Validation:**
  - 5-fold Stratified K-Fold cross-validation
  - Ensures balanced class distribution in each fold
  - Mean accuracy: ~78.7% (±0.28% std)
  
- **Evaluation Metrics:**
  - Confusion matrices for detailed class-wise performance
  - Per-class precision, recall, and F1-scores
  - Feature importance visualization using XGBoost's built-in plot_importance

### 5. Data Preprocessing Pipeline
**Location:** `pre_processing/preprocessing.py`

- **Custom Dataset Class:** PyTorch Dataset implementation for efficient data loading
- **Transformations:**
  - Image resizing to 224×224
  - RGB conversion (even for grayscale images)
  - Tensor conversion and normalization
  - Random shuffling for training
- **Train-Test Split:** 85% training, 15% testing with stratified sampling
- **Batch Processing:** DataLoader with configurable batch size and parallel workers

### 6. Model Evaluation and Visualization
- **Metrics Tracked:**
  - Training and validation loss per epoch
  - Test accuracy across different configurations
  - Confusion matrices for detailed error analysis
  
- **Visualization:**
  - Loss curves for training convergence analysis
  - Saliency maps for model interpretability
  - Generated GAN samples to assess quality
  - Feature importance plots for XGBoost

### 7. Ensemble and Comparison Strategy
The project implements multiple complementary approaches:
- **Deep Learning:** CNNs for end-to-end learning from raw images
- **Traditional ML:** XGBoost with engineered features for interpretability
- **Generative Models:** GANs for data augmentation and exploring data distribution
- **Optimization Comparison:** Systematic evaluation of different optimizers and hyperparameters

This multi-faceted approach allows for comprehensive comparison of techniques and identification of the most effective methods for AD severity classification.

## Results and Key Findings

### Model Performance
- **XGBoost with ResNet50 Features:** ~78.7% test accuracy with low variance (±0.28%)
- **CNN Models:** Performance varied based on architecture and optimization strategy
- **Data Augmentation:** Grid search identified optimal augmentation parameters improving model robustness
- **GAN-Generated Data:** Successfully created synthetic MRI images for data augmentation

### Confusion Matrix Analysis
The overall confusion matrix from XGBoost cross-validation revealed:
- Strong performance on Non-Demented and Mild Demented cases
- Some confusion between adjacent severity levels (expected due to gradual disease progression)
- Class imbalance challenges addressed through augmentation techniques

### Best Practices Identified
1. **Transfer Learning:** Pre-trained ResNet models provided strong feature representations
2. **Feature Extraction:** Deep learning features from ResNet50 outperformed handcrafted texture features
3. **Data Augmentation:** Essential for addressing class imbalance in medical imaging
4. **Cross-Validation:** Stratified K-Fold CV provided robust performance estimates
5. **Ensemble Potential:** Different model types (CNN, XGBoost) capture complementary patterns

## Technical Highlights

### Innovation Points
1. **Comprehensive Comparison:** Side-by-side evaluation of deep learning vs traditional ML approaches
2. **WGAN-GP Implementation:** Advanced GAN architecture with gradient penalty for stable training
3. **Systematic Hyperparameter Search:** Grid search over augmentation parameters and optimizer configurations
4. **Multi-Level Feature Extraction:** Both low-level texture and high-level semantic features
5. **Attention Mechanisms:** Integration of attention layers to focus on diagnostically relevant brain regions

### Dependencies and Tools
- **Deep Learning Frameworks:** PyTorch (CNNs), Keras/TensorFlow (GANs)
- **ML Libraries:** scikit-learn (preprocessing, evaluation), XGBoost (tree models)
- **Image Processing:** PIL, torchvision.transforms, mahotas (texture features)
- **Visualization:** matplotlib, pandas
- **Environment Management:** Conda (Python 3.9)

### Code Organization
- Modular design with separate folders for each technique
- Reusable preprocessing pipeline and custom dataset classes
- Saved models and weights for reproducibility
- Comprehensive logging of training metrics and losses
- Jupyter notebooks for interactive exploration and visualization

## Future Directions

Potential improvements and extensions:
1. **Ensemble Methods:** Combine predictions from CNN and XGBoost models
2. **Advanced Architectures:** Experiment with Vision Transformers (ViT) or EfficientNet
3. **Explainability:** Implement Grad-CAM or LIME for better model interpretability
4. **3D CNNs:** Leverage full 3D MRI volumes instead of 2D slices
5. **Multi-Modal Learning:** Incorporate clinical data alongside imaging
6. **Federated Learning:** Enable privacy-preserving collaborative model training
7. **Longitudinal Analysis:** Track disease progression over time for individual patients

## MIT License
Copyright (c) [2023] [Benjamin Alexander Randoing & Elsa Bismuth & 
Prathibha Alam]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


