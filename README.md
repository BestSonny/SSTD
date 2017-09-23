[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
# Single Shot Text Detector with Regional Attention

## Introduction

**SSTD** is initially described in our [ICCV 2017 paper](https://arxiv.org/abs/1709.00138).


<img src='demo/main.png' width='800'>


If you find it useful in your research, please consider citing:
```
@inproceedings{panhe17singleshot,
      Title   = {Single Shot Text Detector with Regional Attention},
      Author  = {Pan He and Weilin Huang and Tong He and Qile Zhu and Yu Qiao and Xiaolin Li},
      Note    = {In Proceedings of Internatioanl Conference on Computer Vision (ICCV)},
      Year    = {2017}
     }
@inproceedings{liu2016ssd,
      Title   = {{SSD}: Single Shot MultiBox Detector},
      Author  = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      Note    = {Proceedings of European Conference on Computer Vision (ECCV)},
      Year    = {2016}
    }
```

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
  ```Shell
  git clone https://github.com/BestSonny/SSTD.git
  cd SSTD
  ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
  ```Shell
  # Modify Makefile.config according to your Caffe installation.
  cp Makefile.config.example Makefile.config
  make -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional)
  make runtest -j8
