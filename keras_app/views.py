from django.shortcuts import render
from django.views.generic import TemplateView

import numpy as np
import pandas as pd
from io import StringIO, BytesIO
from django.conf import settings
import sys
import os

import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

import base64
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import HttpResponse

# --- コンテキスト
class BaseContext:
    template_name = "common.html"
    def get_context_data(self, **kwargs):
        # テンプレートファイルにparamsデータを渡す
        context = super().get_context_data(**kwargs)
        params = {
            'title': '機械学習 | Django基礎',
            'header': self.header,
            'current_path': self.request.path, # パスの取得
            'val': self.val, # 通常の出力
            'pre': self.pre, # <pre>で出力
            'url': self.url, # 画像出力
        }
        context.update(params)
        return context
    
class IndexView(TemplateView):
    template_name = "index.html"
    # コンテキスト
    header = 'ディープラーニングで画像処理をしよう'
    def get_context_data(self, **kwargs):
        # テンプレートファイルにparamsデータを渡す
        context = super().get_context_data(**kwargs)
        params = {
            'header': self.header,
        }
        context.update(params)
        return context

# --- 4章 TensorFlow・Kerasとは #2 --- #
class Chap04View(BaseContext, TemplateView):
    
    '''
    # 4.3 サンプルデータの取得
    data = tf.keras.datasets.fashion_mnist.load_data()
    # pre = data
    (x_train, y_train), (x_test, y_test) = data

    # 4.4 データの前処理
    # val = y_train, len(y_train), set(y_train)
    y_train = utils.to_categorical(y_train)
    # pre = f"{y_train}"

    y_test = utils.to_categorical(y_test)
    x_train = x_train / 255
    x_test = x_test / 255

    # 4.5 ニューラルネットワーク構造の定義
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # --- Django出力 --- #
    # model.summary()の出力をキャプチャ
    buffer = StringIO()
    # 標準出力を一時的にリダイレクト
    sys.stdout = buffer
    model.summary()
    # 標準出力を元に戻す
    sys.stdout = sys.__stdout__
    # キャプチャした内容を取得
    summary = buffer.getvalue()
    # pre = summary
    # print(summary)  # 確認用
    # --- ここまで --- #

    # 4.6 損失関数、最適化方法、評価指標の設定
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # 4.7 ニューラルネットワークの学習
    model.fit(x_train, y_train, batch_size=32, epochs=5)

    # 4.8 ニューラルネットワークの評価
    model.evaluate(x_test, y_test)

    # 4.9 未知データの予測
    img_path = os.path.join(settings.BASE_DIR, 'keras_app/static/img/unknown_sneaker.png')
    img = load_img(img_path, target_size=(28, 28), color_mode = 'grayscale')
    array = img_to_array(img)
    # pre = f"{array}"
    val = f"{x_train.shape} : {x_test.shape}<br>"

    array = array.reshape((1, 28, 28))
    val += f"{array.shape}<br>"

    array = array / 255
    pre = f"{model.predict(array)}"

    # コンテキスト
    header = '4章 TensorFlow・Kerasとは #2'
    val = val
    pre = pre
    url = 'img/unknown_sneaker.png'
    '''
        # コンテキスト
    header = '4章 TensorFlow・Kerasとは #2'
    val = ''
    pre = ''
    url = ''

# --- 5章 Kerasで手書き数字（MNIST）を判定するニューラルネットワークを構築しよう #1 --- #
class Chap05View(BaseContext, TemplateView):
    '''
    # 5.3 乱数の固定
    utils.set_random_seed(0)

    # 5.4 データの取得
    data = tf.keras.datasets.mnist.load_data()
    # pre = f"{data}"
    (x_train, y_train), (x_test, y_test) = data
    # val = f"{x_train.shape} {y_train.shape} {x_test.shape} {y_test.shape}<br>"
    # val += f"{np.amin(x_train)} {np.amax(x_train)}<br>"
    # val += f"{np.amin(y_train)} {np.amax(y_train)}<br>"
    # val += f"{np.amin(x_test)} {np.amax(x_test)}<br>"
    # val += f"{np.amin(y_test)} {np.amax(y_test)}<br>"

    np.set_printoptions(linewidth=300)
    # pre = f"{x_train[2]}"
    # val += f"{y_train[2]}<br>"
    
    # Matplotlibを使って画像を描画
    fig, ax = plt.subplots()
    ax.imshow(x_train[2], cmap='gray')
    ax.axis('off')

    # 画像をメモリ内に保存
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    # Base64形式に変換
    url = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 5.5 データの前処理
    x_train = x_train / 255
    x_test = x_test / 255
    # pre = f"{x_train[2]}"

    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    val = f"{y_train[2]}"

    # 5.6 ニューラルネットワーク構造の検討
    
    # コンテキスト
    header = '5章 Kerasで手書き数字（MNIST）を判定するニューラルネットワークを構築しよう #1'
    val = val
    pre = ''
    url = url
    '''
    # コンテキスト
    header = '5章 Kerasで手書き数字（MNIST）を判定するニューラルネットワークを構築しよう #1'
    val = ''
    pre = ''
    url = ''

# --- 6章 Kerasで手書き数字（MNIST）を判定するニューラルネットワークを構築しよう #2 --- #
class Chap06View(BaseContext, TemplateView):
    
    # 前回のデータ
    data = tf.keras.datasets.mnist.load_data()
    (x_train, y_train), (x_test, y_test) = data
    np.set_printoptions(linewidth=300)
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)
    
    # 6.3 ニューラルネットワーク構造の定義
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()

    # 6.4 損失関数、最適化方法、評価指標の設定
    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    # 6.5 ニューラルネットワークの学習
    history = model.fit(x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_split=0.1)

    pre = f"{history.history}"
    df = pd.DataFrame(history.history)
    val = df.to_html()

    utils.set_random_seed(0)

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

    history = model.fit(x_train, y_train,
    batch_size=32,
    epochs=7,
    validation_split=0.1)

    # 6.6 ニューラルネットワークの評価
    model.evaluate(x_test, y_test)

    # 6.7 未知データの予測
    img_path = os.path.join(settings.BASE_DIR, 'keras_app/static/img/unknown_mnist.png')
    unknown_img = load_img(img_path, target_size=(28, 28), color_mode = 'grayscale')
    unknown_array = img_to_array(unknown_img)
    unknown_array.shape

    unknown_array = unknown_array.reshape((1, 28, 28))
    unknown_array.shape
    unknown_array = unknown_array / 255

    result = model.predict(unknown_array)

    # Matplotlibを使って画像を描画
    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(data=df[['loss', 'val_loss']], ax=ax)
    # sns.lineplot(data=df[['accuracy', 'val_accuracy']])
    sns.barplot(x=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], y=result[0])
    ax.set_title('barplot')

    # 画像をメモリ内に保存
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plt.close(fig)

    # Base64形式に変換
    url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    sns.lineplot(data=df[['loss', 'val_loss']])
    
    # コンテキスト
    header = '6章 Kerasで手書き数字（MNIST）を判定するニューラルネットワークを構築しよう #2'
    val = val
    pre = pre
    url = url