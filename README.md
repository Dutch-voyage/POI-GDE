# POIGDE

This is the pytorch implementation for paper: 

[**Siamese Learning based on Graph Differential Equation for Next-POI Recommendation**](https://dl.acm.org/doi/10.1016/j.asoc.2023.111086).

Please cite our paper if you use the code.

## Environment Requirement

The code has been tested running under Python 3.8.10. The required packages are as follows:

- pytorch == 1.11.0
- torch-geometric ==  2.1.0
- pandas == 1.5.1
- hydra-core == 1.3.2
- torchsort == 0.1.9

## Running

Here is the process of running the model with NYC dataset.

### 1. unzip dataset

In the main folder, unzip the data package.

~~~
unzip data.zip
~~~

`data.zip` contains the dataset of NYC. You can download dataset of TKY and SG from [here](https://drive.google.com/file/d/1x0ZXpu9SP0xvmdvJOB0wj2VI_t5WS_HT/view?usp=sharing) . 

We filter out the user with interaction times less than 10 and POI visited times less than 10.

### 2. create new folder

In the main folder, create a folder to store running records and another folder to store checkpoint.

~~~
mkdir logs
mkdir ckpts
~~~

### 3. preprocess the dataset

Run process.py to generate data for model from the dataset (operate on NYC by default). 

~~~
python utils/process.py
~~~

### 4. run the model

Run the model with the following command.

~~~
python ode_main.py hydra.job.chdir=False
~~~






