## 예측모델 팀플
* 사용 라이브러리: https://nixtlaverse.nixtla.io/

## 파일 설명

* TSAD_JY[Final_X].ipynb -- Univariate with Exogenous TSAD (Multivariate)
실험 모델: LSTM, GRU, NBEATSx, NHITS, TSMixerx

* TSAD_JY[Final_M].ipynb -- Multivariate TSAD
실험 모델: TSMixer, NHITS, PatchTST, TimesNet

## 주요 인자
* horizon: 예측 길이
* input_size: 학습 길이
* threshold: 이상치 판정 경계값

## RUN CODE
ENV=Ubuntu 20.04 LTS CUDA 12.1.0 cudnn8

`conda create -n TS PYTHON==3.9`
`pip install pip`
`pip install neuralforecast`

Choose what you want :)

`bash TSAD_JY.sh`

OR

If you want to give your own parameters, see below:

`python TSAD_JY_X.py --horizon 10 --input_size 20 threshold 4.0`
`python TSAD_JY_M.py --horizon 20 --input_size 30 threshold 2.0`
