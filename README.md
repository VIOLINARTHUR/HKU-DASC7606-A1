# HKU-DASC7606-A1
HKU DASC 7606 Assignment 1 (Computer Vision: Image Classification), 2024-2025 Fall

This codebase is only for HKU DASC 7606 (2024-2025) course. Please don't upload your answers or this codebase
to any public platforms (e.g., GitHub) before permitted. All rights reserved.

# 1 Introduction
## 1.1 Background: Image Classification
Image classification is a fundamental problem in computer vision, which involves assigning a label or category to an image. The goal is to develop a model that can automatically identify and classify images into different categories. The categories can be objects, actions, or scenes. For example, a model can be trained to classify images of animals, vehicles, or buildings into different categories.Since solutions to image classification problems based on deep learning have become highly mature, the main objective of this assignment is to help you become familiar with the complete workflow of configuring neural networks, including GPU usage, model training and testing, and network design.

## 1.2  What Will You Learn from This Assignment?
This assignment will guide you through the setup and use of a GPU cluster. Following the provided examples, you will learn how to train and test simple neural networks, design network architectures, and more. You will be required to implement and train a convolutional neural network (CNN) on your own. Additionally, you will explore methods to improve network performance, such as by incorporating batch normalization.

The goals of this assignment are as follows:

- Gain experience of implementing neural networks with a popular deep learning framework
PyTorch.
- Develop a deep learning system from scratch, including network design, model training,
hyperparameter tuning, training visualization, model inference and performance evaluation.


## 1.3 Why Image Classification?
**Assignment 1** and **Assignment 2** are both tasks in computer vision, designed with progressively increasing difficulty. In **Assignment 1**, the goal is to train a network that converts images into category labels, which involves extracting abstract semantic information from low-level visual data. In contrast, **Assignment 2** will guide you to train a generative model, where the task is to generate corresponding images from label information. 

From this perspective, category-based generation and image-based classification tasks are fundamentally about the bidirectional alignment of two different modalities of information. Whether generation via denoising or classification via supervision (recognition), both tasks involve processes of **Compression** and **Information Gain**. 

**"Compress what is similar; contrast what is dissimilar."** Starting from **Assignment 1**, you are encouraged to understand this core concept.

# 2 Setup
You can work on the assignment in one of two ways: locally on your own machine, or on a virtual machine on HKU GPU Farm.

## 2.1 Working remotely on HKU GPU Farm (Recommended)
Note: after following these instructions, make sure you go to work on the assignment below (i.e., you can skip the Working locally section).

As part of this course, you can use HKU GPU Farm for your assignments. We recommend you follow the quickstart provided by the [official website](https://www.cs.hku.hk/gpu-farm/quickstart) to get familiar with HKU GPU Farm.

After checking the quickstart document, make sure you have gained the following skills:
- Knowing how to access the GPU Farm and use GPUs in interactive mode. We recommend using GPU support for this assignment, since your training will go much, much faster.
- Getting familiar with running Jupyter Lab without starting a web browser.
- Knowing how to use tmux for unstable network connections.

## 2.2 Working locally on your own machine
If you have the GPU resources on your own PC/laptop and wish to use that, that’s fine – you’ll need to install the drivers for your GPU, install CUDA, install cuDNN, and then install PyTorch. You could theoretically do the entire assignment with no GPUs, though this will make training the model much slower.

## 2.3 Environment Setup
**Installing Python 3.8+**: 
To use python3, make sure to install version 3.8+ on your machine.

**Virtual environment**: The use of a virtual environment via Anaconda is recommended for this project. If you choose not to use a virtual environment, it is up to you to make sure that all dependencies are installed. To establish a Conda virtual environment, execute the following commands:
```bash
git clone https://github.com/VIOLINARTHUR/HKU-DASC7606-A1.git
conda create -n cv_env python=3.10
conda activate cv_env
```
Follow the official PyTorch installation guidelines to set up the PyTorch environment. This guide uses PyTorch version 2.0.1 with CUDA 11.8. Alternate versions may be specified by adjusting the version number:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
Install other requirements:

```bash
pip install matplotlib
pip install tqdm
```

## 3 Working on the assignment
### 3.1 Code & Data
Everything you need to do is provided in the [Jupyter notebook](Assignment_1.ipynb)! **Please make sure to follow the instructions in this [tutorial](https://www.cs.hku.hk/gpu-farm/quickstart#:~:text=Running%20Jupyter%20Lab%20without%20Starting%20a%20Web%20Browser) to set up Jupyter Lab.** Running the example section will automatically download the required data. If you are unable to run the Jupyter notebook, you will need to manually export the code blocks as Python files and organize them accordingly (which is also a great practice opportunity!).

### 3.2 Assignment tasks
**Task 1: Fill in the blank**

There are nine code blocks in the [Jupyter notebook](Assignment_1.ipynb) that require you to complete them.

**Task 2: Write a report (no more than 2 pages)**

Your report should include three main sections: introduction, method, and experiment. See details below.

### 3.3 Files to submit
1. Final Report (PDF, up to 2 pages)

    1.1 Introduction. Briefly introduce the task & background & related works.

    1.2 Methods. Improvements to the baseline model, including but not limited to the methods above.

    1.3 Experiments & Analysis (IMPORTANT) Analysis is the most important part of the report. Possible analysis may include but is not limited to:
    - Ablation studies on validation set. Analyze why better performance can be achieved when you made some modifications, e.g. hyper-parameters, model architectures, and loss functions. The performance on the validation set should be given to validate your claim.
    - More analysis, such as the loss curve. We would not provide the code to save logs with tools such as [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html) or [wandb](https://docs.wandb.ai/guides/integrations/pytorch) for drawing the figure. It is easy to implement and you should find recourses online to enrich your report.
2. Codes

    2.1 All the code files including the `Assignment_1.ipynb` file with model output.

    2.2 README.txt if you added some python files.

3. Model Weights

    Models, in the format of model checkpoint link (model_link.txt) due to the limitation on submission file size.
    Please ensure adherence to model naming conventions and ensure compatibility with the code.

If your student id is 30300xxxxx, then the compressed file for submission on Moodle should be organized as follows:
```
30300xxxxx.zip
├── report.pdf
├── your code (Must include the Assignment_1.ipynb file)
├── model_link.txt
└── (optional) README.md
```
### 3.4 Timeline
September 12, 2024 (Thu.): The assignment release.  
October 8, 2024 (Tue.): Submission deadline (23:59 GMT+8).

Late submission policy:

- 10% for late assignments submitted within 1 day late. 
- 20% for late assignments submitted within 2 days late.
- 50% for late assignments submitted within 7 days late.
- 100% for late assignments submitted after 7 days late.

### 3.5 Need More Support?
For any questions about the assignment which potentially are common to all students, your shall first look for related resources as follows,
- We encourage you to use [GitHub Issues](https://github.com/VIOLINARTHUR/HKU-DASC7606-A1/issues) of this repository.
- Or if you prefer online doc: [Discussion doc](https://connecthkuhk-my.sharepoint.com/:w:/g/personal/u3011175_connect_hku_hk/EWxAcIs50gJMlN-zDnC6qvkBsRbyrIa6GLX1WnlduuBfhA?e=oPsX4J).

For any other private questions, please contact Tianshuo Yang (yangtianshuo@connect.hku.hk) via email.

## 4 Marking Scheme:
Marks will be given based on the performance that you achieve on the test and the submitted report file. TAs will perform an evaluation of the model predictions.

The evaluation criteria are divided into two primary components: (1) model performance on the test datasets, and (2) the quality of the final report, with the latter accounting for 20% of the total marks:
1. Coding and Model Performance (80% of total marks):

    Marks will be given based on your coding and the performance of your model. 
    - For the coding part, you need to complete all the required codes in order to get the full score. Partial score will be considered rarely and carefully.
    - For the performance part, the mark will be given based on the accuracy of your result on test set.
        - Accuracy above 80% will get the full mark of this part.
        - Accuracy between 70-80% will get 90% mark of this part.
        - Accuracy between 65-70% will get 80% mark of this part.
        - Accuracy between 60-65% will get 70% mark of this part.
        - Accuracy between 50-60% will get 60% mark of this part.
        - Others will get 0% mark.

2. Final Report (20% of total marks): 

    The marks will be given mainly based on the richness of the experiments & analysis.
    - Reasonable number of experiments + analysis: 90%-100% mark of this part.
    - Basic analysis: 80%-90% mark of this part.
    - Not sufficient analysis: lower than 80%.

## Reference
1. ImageNet Classification with Deep Convolutional Neural Networks. NIPS 2012: [pdf](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
2. Deep Residual Learning for Image Recognition. CVPR 2016: [pdf](https://arxiv.org/pdf/1512.03385)
3. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML 2015: [pdf](https://arxiv.org/pdf/1502.03167)
