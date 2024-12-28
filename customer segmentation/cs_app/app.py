from flask import *
import pickle
import joblib
import numpy as np

app=Flask(__name__)

model = pickle.load(open("csmodel.pkl", "rb"))
scaler = joblib.load('ssscaler.pkl')

@app.route('/', methods=["GET", "POST"])
def home():
	if request.method == "POST":
		age=int(request.form["age"])
		ai = int(request.form["ai"])
		sc= int(request.form["sc"])
		sex = int(request.form["sex"])
		if sex == 0:
			d = [[age, ai, sc, 0,1]]
		else:
			d = [[age, ai, sc, 1,0]]
		
		print(age,ai,sc,sex)
		nd = scaler.transform(d)
		res = model.predict(nd)
		res = res[0]
		print(res)
		if res == 0:
			ans = "C0: min income min spending"
		elif res ==1:
			ans = "C1: high income low spending"
		elif res == 2:
			ans = "C2: high income high spending"
		elif res == 3:
			ans = "C3: low income low spending"
		else:
			ans = "C4: low income high spending"

		return render_template("home.html", msg=ans)
		
	else:
		return render_template("home.html")

if __name__=='__main__':
    app.run(debug=True, use_reloader=True)             