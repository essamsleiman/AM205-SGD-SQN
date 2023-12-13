# AM205-SGD-SQN

Run these scripts to train different ML models on the wine-quality dataset with the three optimizers

<!--  model ML1 is the convex problem -->

1. python3 run_ml.py --model ML1 --epochs 10 --num_features 11 --batch_size 100

<!--  model ML3 is the non-convex problem -->

2. python3 run_ml.py --model ML3 --epochs 10 --num_features 11 --batch_size 100

Run this script to get the minimum eigenvalue of the Hessian for the different ML models when trained with RES and oBFGS optimizers on the wine-quality dataset

<!--  model ML1 is the convex problem -->

1. python3 get_min_eig.py --model ML1 --epochs 10 --num_features 11 --batch_size 100

<!--  model ML3 is the nonconvex problem, however this script will fail -->

1. python3 get_min_eig.py --model ML3 --epochs 10 --num_features 11 --batch_size 100

# AM205-SGD-SQN

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
