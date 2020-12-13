import pickle
from flask import Flask, render_template, request, url_for, redirect
import numpy as np


filename = 'model.pkl'
model = pickle.load(open(filename,'rb'))
le=pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    team1 = str(request.args.get('list1'))
    team2 = str(request.args.get('list2'))

    toss_win = int(request.args.get('toss_winner'))
    bating = int(request.args.get('Batting'))

    # with open('vocab.pkl', 'rb') as f:
    #     vocab = pkl.load(f)
    # with open('inv_vocab.pkl', 'rb') as f:
    #     inv_vocab = pkl.load(f)

    # with open('model.pkl', 'rb') as f:
    #     model = pkl.load(f)

    cteam1 = le.transform([team1])
    cteam2 = le.transform([team2])
    if toss_win==0:
        toss=le.transform([team1])
    else:
        toss=le.transform([team2])

    if bating==0:
        first_bat = le.transform([team1])
    else:
        first_bat = le.transform([team2])

    if cteam1 == cteam2:
        return redirect(url_for('index'))

    lst = np.array([cteam1, cteam2, first_bat, toss], dtype='int32').reshape(1,-1)

    prediction = model.predict_proba(lst)

    if prediction[0,cteam1] > prediction[0,cteam2]:
        team_win = team1

    else:
        team_win = team2

    return render_template('predict.html', data=team_win)
    



if __name__ == "__main__":
    app.run(debug=True)
