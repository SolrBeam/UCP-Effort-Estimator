from __future__ import print_function

import flask
import numpy as np

from sklearn.externals import joblib
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = flask.Flask(__name__)

model = joblib.load('model.pkl')

def getitem(obj, item, default):
	if item not in obj:
		return default
	else:
		return obj[item]

def prediction(uaw, uucw, tf, ef):
	uucp = uaw + uucw
	ucp = uucp * tf * ef
	estimation = ucp * 20
	
	return estimation

@app.route("/")
def user_request():
	args = flask.request.args

	uaw = int(getitem(args, 'uaw', 0))
	uucw = int(getitem(args, 'uucw', 0))
	tf = int(getitem(args, 'tf', 0))
	ef = int(getitem(args, 'ef', 0))

	estimation = prediction(uaw, uucw, tf, ef)
	real = model.predict(np.array([[estimation]]))

	js_resources = INLINE.render_js()
	css_resources = INLINE.render_css()

	html = flask.render_template(
		'index.html',
		js_resources=js_resources,
		css_resources=css_resources,
		uaw=uaw,
		uucw=uucw,
		tf=tf,
		ef=ef,
		estimation=estimation,
		real=int(round(real[0]))
	)
	return encode_utf8(html)

if __name__ == "__main__":
	print(__doc__)
app.run()