# MNIST Neural Network

A feedforward neural network implemented in Python using NumPy trained on MNIST

## Setup & Configuration

### 1. Environment Setup

#### macOS / Linux

```bash
# go to project folder
cd /path/to/project/

# create virtual environment
python -m venv .venv

# activate virtual environment
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [MNIST as JPG dataset on Kaggle](https://www.kaggle.com/datasets/scolianni/mnistasjpg?select=testSet)

Organize the dataset like this:
```
dataset/
├── test/
│   ├── *.jpg
│
└── train/
    ├── 0/
    ├── 1/
    ├── 2/
    ├── 3/
    ├── 4/
    ├── 5/
    ├── 6/
    ├── 7/
    ├── 8/
    └── 9/
```


### 3. Configure  Network (Optional)

Before running, you can customize the network

Open **`src/trainer.py`** 

#### Configure Random Seed
Locate `np.random.seed(42)`:

* Change the number to get different initializations

#### Configure Layer Sizes
Locate the `layer_sizes` list:

```python
layer_sizes = [784, 128, 64, 10]
```

* *Input:* Set to **`784`** to match the $28 \times 28$ pixel images from the MNIST database

* *Output:* Set to **`10`** to match the 10 target classes

* *Hidden Layers:* Add, remove, or change these freely

#### Configure Training
 
Locate the `.train()` line at the bottom:

```python
net.train(X_train, Y_train, epochs=200, learning_rate=0.1, plot=True)
```

* Change `epochs` to change how many times the network loops through the data

* Change `learning_rate` to make steps larger or smaller

## Quick Start

### 1. Train Network

Saves model to `models/network.npz`:

```bash
python src/trainer.py
```

### 2. Test Prediction

Loads model from `models/network.npz` 

Open **`src/tester.py`** 

#### Configure Test Image

Locate `IMAGE_PATH = "dataset/test/img_1.jpg"`

* Change `img_1.jpg` to any image from `dataset/test/*.jpg`

```bash
python src/tester.py
```

### 3. Exit Virtual Environment

```bash
deactivate
```