from flask import render_template, Flask, request
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('twitter.h5')
cla.compile(optimizer='adam',loss='binary_crossentropy')
app = Flask(__name__)
@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')
@app.route('/tpredict', methods = ['GET', 'POST'])
def page2():
    topic = request.form['tweet']
    print("Hey " +topic)
    topic=cv.transform([topic])
    print("\n"+str(topic.shape)+"\n")
    with graph.as_default():
        y_pred = cla.predict(topic)
        print("pred is "+str(y_pred))
    if(y_pred > 0.5):
        topic = "Positive Tweet"
    else:
        topic = "Negative Tweet"
    return topic
if __name__ == '__main__':
    app.run(host = 'localhost', debug = True , threaded = False)
