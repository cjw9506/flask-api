from flask import Flask, render_template, request
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from konlpy.tag import Kkma
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

def morphemes(): #입력받은 값을 명사, 동사만 뽑아서 리스트화
    content = request.form['id_name']

    kkma = Kkma()
    a, b = kkma.nouns(content), []
    
    if content == " " or "":
        pass
    else:
        kma = kkma.pos(content)
        for x in kma:
            for word, tag in enumerate(x):
                if tag in ['VV']:
                    b.append(x[0])

    word = " ".join(a+b)
    words = word.split(maxsplit=0)
    return words

def read_data(): #xlsx파일 읽어서 전처리하기전에 마지막 인덱스에 입력한 값 추가
    words = morphemes()
    df = pd.read_excel('crawling.xlsx')
    
    arr = df['명사+동사'].values
    doc = arr.tolist()
    docs = doc + words
    
    tokenizer = Tokenizer(num_words = 10000)
    tokenizer.fit_on_texts(docs)
    sequences = tokenizer.texts_to_sequences(docs)
    data = pad_sequences(sequences, maxlen=300)
    return data

def result(): #모델 돌려서 prediction 기준으로 리턴값 준것
    category = read_data() 
    model = load_model('dp__model.h5')
    predictions = model.predict(category)
    a = predictions[-1]
    a = a.tolist()
    #return str(a[0])+" "+str(a[1])+" "+str(a[2])+" "+str(a[3])
    b = 0
    for x in a:
        if x >= 0.8:
            b += 1
    if b == 0:
        return "전체메일함"
    else:
        if np.argmax(predictions[-1]) == 0:
            return "광고"
        elif np.argmax(predictions[-1]) == 1:
            return "구매 및 주문내역"
        elif np.argmax(predictions[-1]) == 2:
            return "로그인 및 인증내역"
        elif np.argmax(predictions[-1]) == 3:
            return "교육"
    
@app.route('/', methods=['GET', 'POST'])
def test(): #이부분은 너가 더 잘 알거라고 생각쓰..
    if request.method =='GET':
        return render_template('post.html')
    elif request.method == 'POST':
        val1 = request.form['sender']
        val2 = request.form['title']
        val3 = request.form['id_name']
        val4 = result()
        return render_template('default.html', data1=val1,data2=val2,data3=val3,data4=val4)

if __name__ == "__main__":
	app.run()