# An Examination of Stochastic Quasi-Newton Methods for Optimization in Multiple Dimensions

## Introduction

Stochastic gradient descent (SGD) methods comprise one of the most popular families of algorithms for addressing large-scale optimization and machine-learning problems. These methods rely on the gradient of the cost function to update parameters toward an optimal value and are thus informed by first-order information. While SGD algorithms have proven to be a versatile tool for many applications, they are nonetheless prone to performance issues for highly nonlinear or ill-conditioned problems due to a lack of scale invariance.

As optimization methods incorporating second-order information, i.e. the Hessian or an approximation thereof, have shown performance improvements over gradient-based methods in the deterministic case, researchers have recently sought to answer whether analogous benefits may be obtained for their stochastic analogs.

In this paper, we compare the performance of stochastic second-order methods to SGD and its variants, focusing on the family of stochastic quasi-Newton (SQN) methods. We begin by formally outlining the problem setting, describing how stochastic optimization methods arise as a technique for optimizing separable loss functions, which arise naturally when the objective is the expected value of a random variable. We also introduce SGD as one of the most popular approaches for addressing this type of problem and outline the limitations that motivate the exploration of second-order methods. Next, we describe the relevant theory underlying SQN methods, starting from the deterministic case and ending with the presentation of two SQN algorithms. In the following section, we specify the loss functions that we will investigate, and then, we present the results of our numerical experiments. Finally, we conclude with a summary and outlook.


## Description

This project contains scripts to train different machine learning models on the wine-quality dataset using three different optimizers. Additionally, it provides a script to compute the minimum eigenvalue of the Hessian for the trained models when using specific optimizers.

## Usage

### Training Machine Learning Models

#### Model ML1 (Convex Problem)

To train the ML1 model, which represents the convex problem, use the following command:

```bash
python3 run_ml.py --model ML1 --epochs 10 --num_features 11 --batch_size 100
```

#### Model ML3 (NON-Convex Problem)

To train the ML3 model, which represents the non-convex problem, use the following command:

```bash
python3 run_ml.py --model ML3 --epochs 10 --num_features 11 --batch_size 100
```

### Computing Minimum Eigenvalue of Hessian

To compute the minimum eigenvalue of the Hessian for different ML models when trained with RES and oBFGS optimizers on the wine-quality dataset, use the following commands:

#### Model ML1 (Convex Problem)

```bash
python3 get_min_eig.py --model ML1 --epochs 10 --num_features 11 --batch_size 100
```

#### Model ML3 (NonConvex Problem)

```bash
python3 get_min_eig.py --model ML3 --epochs 10 --num_features 11 --batch_size 100
```

### Data

You can download the dataset from the following link and place it in the current directory:

[Download Wine Quality Dataset](https://www.kaggle.com/code/abdelruhmanessam/wine-quality)

## License

This project is licensed under the [MIT License](LICENSE).
