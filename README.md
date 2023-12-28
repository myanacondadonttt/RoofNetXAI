# Abstract

This master thesis explores the intersection of accuracy and explainability in machine learning models, with a focus on solar panel installation applications. The research introduces a novel approach by integrating explainable artificial intelligence (XAI) techniques, specifically SHapley Additive explanations (SHAP), into the training process. Two innovative regularization methods, SHAP Deviation Regularization (SDR) and SHAP Intensity Regularization (SIR), are developed to balance accuracy and explainability, enhancing model reliability. Additionally, the thesis utilizes average sensitivity as a new metric to quantitatively assess model explainability. The results demonstrate that XAI-enhanced models show improved performance in imbalanced datasets, particularly under noisy conditions, and exhibit more stable and consistent explanations, highlighting the potential of XAI in advancing machine learning model performance and explainability.


# Use case: RoofNet
Deep learning project to identify roof age using historical satellite images to lower the customer acquisition cost for new solar installations. 

Project in conjuction with Misha Laskin, Chris Heinrich, and Evert P.L. van Nieuwenburg

Paper available: https://arxiv.org/abs/2001.04227

Rooftop solar represents a crucial solution in reducing greenhouse gas (GHG) emissions, offering a cost-effective alternative to fossil fuels in various regions worldwide. A key factor in assessing the feasibility of rooftop solar installations is the age of the building's roof. This is critical because the lifespan of the roof should align with the longevity of the solar array to avoid premature replacements. This study introduces a data-driven approach to estimate roof age from historical satellite imagery, facilitating an automated pipeline for rooftop solar site selection. The method utilizes a convolutional variational autoencoder combined with a deep fully connected binary classifier, trained on historical satellite images of rooftops. Labels for training are obtained from public reroof permit data. During inference, images are encoded into latent vectors by the variational autoencoder and concatenated for classification. The age of the roof is determined by finding the argmax over temporally adjacent images, with roof age considered undefined if no pair of images classify as a reroof.


# Notebooks

Code is represented in following notebooks:

- Main SDR Average Training, 10 runs - 1 Model
- Main SOTA, SDR K-fold, SIR Training - 3 Models 
- Main Evaluation - Evaluation Notebook for Robustness of a Model, Average Sensitivity of an Explanation

One pre-trained VAE is used for every model.
