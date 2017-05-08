from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')
	
@app.route("/input")
def input():
    return render_template('input.html')
	
@app.route("/view")
def view():
    return render_template('view.html')

if __name__ == "__main__":

	try:
		clf = joblib.load('model.pkl')
	
	except Exception, e:
		print 'No Model Here'
	
	app.run()