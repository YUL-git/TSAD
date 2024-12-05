## Predictive Modeling Team Project
* Libraries Used: https://nixtlaverse.nixtla.io/

## File Description

* TSAD_JY[Final_X].ipynb -- Univariate with Exogenous TSAD (Multivariate)  
Experimental Models: LSTM, GRU, NBEATSx, NHITS, TSMixerx

* TSAD_JY[Final_M].ipynb -- Multivariate TSAD  
Experimental Models: TSMixer, NHITS, PatchTST, TimesNet

## Key Parameters    
* horizon: Prediction length
* input_size: Training window size
* threshold: Anomaly detection threshold

## RUN CODE
ENVIRONMENT SET  
Ubuntu 20.04 LTS CUDA 12.1.0 cudnn8

```
conda create -n TS PYTHON==3.9
pip install pip
pip install neuralforecast
```
Choose what you want :)

```
bash TSAD_JY.sh
```

OR

If you want to give your own parameters, see below:

```
python TSAD_JY_X.py --horizon 10 --input_size 20 threshold 4.0
python TSAD_JY_M.py --horizon 20 --input_size 30 threshold 2.0
```
