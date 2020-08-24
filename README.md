# ml-project-template

機械学習の実験やプロジェクトの管理方法に関するメモやノウハウをまとめる．
SIGNATEの練習問題を解きながら実際の開発環境を構築

## 研究の進め方
研究の流れとともにそこで利用されるであろうツールを整理
### 1. データを見る．

jupyterが基本．実験しながら方針を固めていく．全てのデータに目を通し，人間である自分ならどれくらいの認識が可能かを見定める．
また，データの可視化(PCA,t-SNE,UMAPあたり)してみて，どういうモデルならできそうか(画像であればCNN, NNを使うまでもなければどんな特徴量が必要か)をあたりをつける．ここで全てが決まると思っても良い．全体の50%の時間をかける
### 2. ベースモデルの策定

データを観察することがだいたい終了したら，モデルを決める．
デファクトスタンダードになっている手法(これであれば〇〇%くらい精度がでそうとあたりがつくもの)を利用する．
ここで使うモデルによって前処理方法も異なるはず

### 3. 前処理

データの前処理を行う．Deep系ならAugmentationなども決めておく．



### 4. 学習
### 5. 予測



## ディレクトリ構造


## ユーザ
$whoami
$id your_name

user: "${UID}:${GID}"
UID=${UID} GID=${GID} docker-compose up
UID=${UID} GID=${GID} docker-compose exec python-gpu bash

## How to get start?
### 1. Connect to remote server
```shell
$ssh -l user@ip
```
### 2. prepare for tmux terminal
you don't have to reconncet to new server when you are session out
```shell
$tmux
```
### 3. build docker containerls

```shell
# build containers
$UID=your_uid GID=your_gid docker-copmose build
# start containers
$UID=your_uid GID=your_gid docker-compose up
# exec containers
$UID=your_uid GID=your_Gid docker-compose exec python-gpu bash
```

### 4. reconnect to tmux

```shell
# 1. connec to remote
$ssh
# 2. check tmux session on remote server
$tmux ls
# 3. connect to session
$tmux a -t <sessionid>
```
Please check references for any other usages


## tips
- データはgitで管理しない．容量制限で大抵upできないから
- Dataフォルダは生データ，処理済み，データセット(訓練検証)用で構成
- パラメータ管理はhydraを使用
- 実験結果の管理はmflow or sacredを使用
- 環境はdockerで管理

## Data Leakageについて

https://towardsdatascience.com/data-leakage-in-machine-learning-10bdd3eec742
A very common error that people make is to leak information in the data pre-processing step of machine learning. It is essential that these transformations only have knowledge of the training set, even though they are applied to the test set as well. For example, if you decide that you want to run PCA as a pre-processing step, you should fit your PCA model on only the training set. Then, to apply it to your test set, you would only call its transform method (in the case of a scikit-learn model) on the test set. If, instead, you fit your pre-processor on the entire data-set, you will leak information from the test set, since the parameters of the pre-processing model will be fitted with knowledge of the test set.
## 参考
SIGNATEの問題はこちら
https://signate.jp/competitions/108/data

ディレクトリ構成はこちらを参考
https://drivendata.github.io/cookiecutter-data-science/

Deep Learnningのモデルの作成・学習時の実験管理を楽にするツールたち
https://qiita.com/TatsuyaShirakawa/items/db2c37ab21df109e72ab

情報系研究者のための研究ノート
https://qiita.com/guicho271828/items/9307ae12248329b71f12

データセットの作り方についてはこちらを参考
https://betashort-lab.com/%E3%83%87%E3%83%BC%E3%82%BF%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9/%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0/pytorch%E3%81%AEdatasets%E3%81%A7%E7%94%BB%E5%83%8F%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88%E3%82%92%E4%BD%9C%E3%82%8B/

https://qiita.com/daikiclimate/items/26be9bb74f1c27c54d23

MLFlowに関する使い方は
https://www.slideshare.net/maropu0804/mlflow

MLFlow × Dockerに関しては
https://github.com/hyzhak/mlflow-container

Best Practices for Hyperparameter Tuning with MLflow
https://www.slideshare.net/databricks/best-practices-for-hyperparameter-tuning-with-mlflow

tmuxの使い方
https://qiita.com/toshihirock/items/77bd3e09abde3bb26067

A Quick and Easy Guide to tmux
https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/


Simple Introduction to Tensorboard Embedding Visualisation
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/

tensorflowのembeddingの例
http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/


## アンチパターン集
Why your machine learning project will fail
http://thedatascience.ninja/2018/07/12/why-your-machine-learning-project-will-fail/


機械学習プロジェクトが失敗する9つの理由
https://tjo.hatenablog.com/entry/2018/08/03/080000
