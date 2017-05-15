from flask import Flask, request, jsonify, render_template
from sklearn.externals import joblib

app = Flask(__name__)

clf = None;

@app.route('/')
def home():
    return render_template('index.html')
	
@app.route('/input', methods=['GET', 'POST'])
def input(clf = None):
	
	if request.method == 'POST':
		uaw = request.form['txtUAW']
		uucw = request.form['txtUUCW']
		tf = request.form['txtTF']
		ef = request.form['txtEF']
		
		est = prediction(uaw,uucw,tf,ef)
		
		clf.predict(est)
		
	else:
		print 'Error Model'
		
    return render_template('input.html')
	
@app.route('/view')
def view():
    return render_template('view.html')

if __name__ == "__main__":

	try:
		clf = joblib.load('model.pkl')
	
	except Exception, e:
		print 'No Model Here'
	
	app.run()
	
def prediction(uaw,uucw,tf,ef):
	
	uucp = uaw + uucw
	ucp = uucp * tf * ef
	estimation = ucp * 20
	
	return estimation