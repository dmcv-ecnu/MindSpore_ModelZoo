# Boosting Night-time Scene Parsing with <br /> Learnable Frequency

This repo is the Mindspore implementation of ["Boosting Night-time Scene Parsing with Learnable Frequency (IEEE TIP 2023)
"](https://ieeexplore.ieee.org/document/10105211).


## Data Preparation

["NightCity"](https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html)

["NightCity+"](https://drive.google.com/file/d/1EDhWx-fcS7pIIBGbu3TpebNrmyE08KzC/view) (Only reannotated val set)

["BDD100K-night"](https://drive.google.com/file/d/1l4Mh3V7OcCbD6GpxPzovloLlRWSAZ4vZ/view?usp=share_link) (Only images, please download the labels from [here](https://bdd-data.berkeley.edu/) with permission)

## Train
```
cd scripts
python train_edge.py
```

## Test
```
cd scripts
python eval.py
```

## Results

| Dataset | w/ ms 
| :---: | :---: |
| NightCity | 48.91  |