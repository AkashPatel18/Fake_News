from flask import Flask,render_template,request
import re
import nltk
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
import pickle
stopword  = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such']
#nltk.download()

tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('fakenews.pkl','rb'))


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict' ,methods = ['POST'])
def predict():
    if request.method == "POST":
        rawtext = request.form['title'] + " " + request.form['author'] + " " + request.form['maintext']

        corpus = []
        review = re.sub('[^a-zA-Z]',' ',rawtext)
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(word) for word in review if not word in stopword]
        review = ' '.join(review)
        corpus.append(review)


        vector = tfidf.transform(corpus).toarray()
        prediction = model.predict(vector)
        


    return render_template('index.html',result = prediction)



if __name__ == "__main__":
    app.run(debug=True)
