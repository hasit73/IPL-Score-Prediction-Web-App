from flask import Flask,request,render_template
import joblib
import os
import numpy as np
app = Flask(__name__)


list_of_stadiums = []
list_of_teams = []
stadium_dict ={}
team_dict = {}
model = None

@app.route('/', methods=['POST','GET'])
def home():
    score = 0
    if request.method=='POST':
        stad = request.form['stadium']
        bat = request.form['batteam']
        bowl = request.form['bowlteam']
        wickets = request.form['wickets']
        overs = request.form['overs']
        runin5  =request.form['runinlast5overs']
        wicketin5 = request.form['wicketinlast5overs']

        pre = model.predict(np.array([[int(stadium_dict[stad]),int(team_dict[bat]),int(team_dict[bowl]),int(wickets),float(overs),int(runin5),int(wicketin5)]]))
        score = round(pre[0][0])
    
    
    return render_template('home.html',stadiums = list_of_stadiums,teams=list_of_teams,num_stadiums=len(list_of_stadiums),num_teams = len(list_of_teams),score = score)


def initiate_server():
    print("File loading....")
    print(os.getcwd())
    global team_dict
    team_dict = joblib.load('teamDict.pkl')
    global stadium_dict
    stadium_dict = joblib.load('stadiumDict.pkl')
    global list_of_stadiums
    list_of_stadiums = list(stadium_dict.keys())
    global list_of_teams
    list_of_teams = list(team_dict.keys())
    global model
    model = joblib.load('linearModelforIPL2.pkl')



print("flask server started")
initiate_server()
app.run(debug=True,use_reloader=False)

