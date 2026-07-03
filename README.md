# BWOM-R2DPCA

This repository provides the implementation of **BWOM-R2DPCA**, a robust two-dimensional principal component analysis method based on **binary weighting** and **optimal mean estimation**.

The proposed method aims to improve the robustness of 2DPCA under corrupted training samples by integrating binary sample weighting, optimal mean estimation, and an L1-norm-based greedy optimization strategy.

For more details, please refer to the paper:

**Robust Two-Dimensional Principal Component Analysis via Binary Weighting and Optimal Mean Estimation**

------

## 1. Repository Structure

```text
BWOM_R2DPCA/
│
├── code/
│   └── BWOM_R2DPCA.py
│
├── datasets/
│   ├── ETH-80/
│   ├── NEC/
│   ├── COIL-100/
│   ├── ORL/
│   ├── GT/
│   └── PIE/
│
├── README.md
└── requirements.txt
```

The main algorithm and experimental code are included in:

```text
BWOM_R2DPCA/code/BWOM_R2DPCA.py
```

------

## 2. System Requirements

The experiments were conducted under the following environment:

```text
Operating System: Windows 10
Python Version: 3.8.16
Processor: Intel(R) Xeon(R) Platinum 8260M CPU @ 2.30 GHz
Memory: 32 GB
```

Required Python libraries:

```text
numpy
opencv-python
scikit-learn
```

The required packages can be installed by:

```bash
pip install -r requirements.txt
```

A simple `requirements.txt` file is provided as:

```text
numpy
opencv-python
scikit-learn
```

------

## 3. Dataset Description

This project uses six image datasets:

```text
ETH-80
NEC
COIL-100
ORL
GT
PIE
```

Each dataset folder is organized as follows:

```text
DatasetName/
│
├── 20/
│   ├── train1/
│   ├── train2/
│   ├── train3/
│   ├── train4/
│   └── train5/
│
├── 40/
│   ├── train1/
│   ├── train2/
│   ├── train3/
│   ├── train4/
│   └── train5/
│
├── 60/
│   ├── train1/
│   ├── train2/
│   ├── train3/
│   ├── train4/
│   └── train5/
│
├── train/
├── test/
├── train.txt
└── test.txt
```

The folders `20`, `40`, and `60` contain training samples with 20%, 40%, and 60% random block occlusion, respectively.

The subfolders `train1` to `train5` correspond to five repeated occlusion experiments.

The folders `train` and `test` contain clean training and testing samples.

The files `train.txt` and `test.txt` provide the corresponding label information for classification evaluation.

------

## 4. Running the Code

The main code can be executed by running:

```bash
python BWOM_R2DPCA.py
```

Before running the code, please configure the dataset paths and dataset parameters in the script.

For example, to evaluate the model on the COIL-100 dataset under 20% occlusion, the paths can be configured as:

```python
train_path = "D:\\DataSet\\...\\COIL-100\\20\\train1\\%d.png"
test_path = "D:\\DataSet\\...\\COIL-100\\test\\%d.png"

trainlable = "D:\\DataSet\\...\\COIL-100\\train.txt"
testlable = "D:\\DataSet\\...\\COIL-100\\test.txt"
```

The dataset parameters can be configured as:

```python
num_train = 700
num_test = 700

m = 100  # number of image rows
n = 100  # number of image columns
```

After running the code, the program outputs the classification accuracy and reconstruction error.

------

## 5. Notes

- The current implementation is written for clarity and reproducibility.
- The code is CPU-based and does not require GPU acceleration.
- The running time may be higher than standard 2DPCA methods because BWOM-R2DPCA uses nested iterative optimization.
- The main focus of this implementation is robust feature extraction under corrupted training samples.

------

## 6. Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{BWOMR2DPCA,
  title   = {Robust Two-Dimensional Principal Component Analysis via Binary Weighting and Optimal Mean Estimation},
  author  = {},
  journal = {The Visual Computer},
  year    = {2026}
}
```

------

## 7. License

This project is released for academic research purposes.

------

## 8. Contact

If you have any questions about the code or experiments, please contact the author or the corresponding investigator.
