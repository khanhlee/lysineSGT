# Reference: https://github.com/krishnaik06/Deployment-flask

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #for deployment purpose - PCA and SGT fitted on training set first
    import pandas as pd
    from sgt import Sgt
    from sklearn.decomposition import PCA
    from pandas import DataFrame
    
    df3 = pd.read_csv (r'kstringclean.csv')
    
    #to create sequence of different length (test len 21 is optimal so far)
    difflen=[]
    for i in range (0,len(df3['Seq original (27)'])):
        text1=df3['Seq original (27)'][i][:-3]
        text1=text1[3:]
        difflen.append(text1)
        
    X=difflen
    
    def split(word): 
        return [char for char in word] 
    
    sequences = [split(x) for x in X]
    
    #Generating sequence embeddings on train set
    sgt = Sgt(kappa = 20, lengthsensitive = True)
    embedding = sgt.fit_transform(corpus=sequences)
    
    #perform PCA on the sequence embeddings
    pca = PCA(n_components=120) # can try 60 to 120 (80 is the best)
    pca.fit(embedding)


    int_features = [int(x) for x in request.form.values()]
    s = int_features[0]
    
    #s = 'MTRILTAFKVVRTLKTGFGFTNVTA'
    s.capitalize()
    
    def charposition(string, char):
        pos = [] #list to store positions for each 'char' in 'string'
        for n in range(len(string)):
            if string[n] == char:
                pos.append(n)
        return pos
    
    def noncharposition(string, neglist):
        neg = [] #list to store positions for each 'char' in 'string'
        for n in range(len(string)):
                if string[n] not in neglist:
                    neg.append(string[n])
        return neg
    
    charp=charposition(s, 'K')
    
    amino_list =['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V'] 
    notfound = noncharposition(s,amino_list)
    
    if len(charp)==0:
        print('This sequence have no K amino acid. The model is not able to predict lysine PTM.')
        
    elif len(notfound)!=0:
        print ("Please check sequence. Model not able to recognise amino acid found in string:", notfound )
    
    else:
        test_sequences=[]
    
        for p in range(0,len(charp)):
            if charp[p]<11:
                startstr = s[0]*(11-charp[p])+s[0:charp[p]-1]
            else:
                startstr = s[charp[p]-11:charp[p]-1]
    
    
            endstr=s[charp[p]:charp[p]+11]
    
            if len(endstr)<12:
                endstr=s[charp[p]:charp[p]+11]+s[charp[p]]*(11-len(endstr))
    
            test_sequences.append(startstr+endstr)
            #endstr =if(len(endstr)!=11,startstr[-1]*(11-len(startstr)))
    
    
        #to create SGT features
        Xtest_sequences = [split(x) for x in test_sequences]
        Xtest_embedding = sgt.fit_transform(corpus=Xtest_sequences)
        Xtest=pca.transform(Xtest_embedding)
    
        model_predy = []
    
        from keras.models import load_model
    
        # load model
        model = load_model('sgt_Fold0.h5', compile=False)
    
        y_test_pred = model.predict_proba(Xtest)
        y_test_pred = y_test_pred.round()
        y_test_pred = y_test_pred.astype(int)
        labels_test_pred = (y_test_pred>0.5).astype(np.int)
        model_predy.append(labels_test_pred)
    
        y_predlist=[]
        for i in range (0,len(model_predy)):
            for k in range (0,len(model_predy[i])):
                y_predlist.append(model_predy[i][k])
    
    
        y_pred_df=DataFrame(y_predlist)
        y_pred_df.columns = ['S1','S2','S3','S4'] 
        y_pred_df['Sequence']=test_sequences
        y_pred_df['Kposition']=charp

    return render_template('index.html', prediction_text='Availability for S1, S2, S3, S4 would be {}'.format(y_pred_df))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)