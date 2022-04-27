from contextlib import AsyncExitStack
from flask import Flask,render_template,request,redirect
import numpy as np
import librosa
import pickle

model = pickle.load(open('modellgbm.pkl','rb'))
print(model)
class SpeechEmotionRecognition:
  def __init__(self,filename,n_mfcc):
    self.filename=filename
    self.n_mfcc=n_mfcc

  def ExtractFeatures(self):
    self.y, self.samplingrate = librosa.load(self.filename, duration=3, offset=0.5)
    self.mfcc = np.mean(librosa.feature.mfcc(y=self.y, sr=self.samplingrate, n_mfcc=self.n_mfcc).T, axis=0)
    self.mfcc=np.array(self.mfcc)
app=Flask(__name__)
@app.route("/",methods=["GET", "POST"])
def hello():
    ans=''
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            print(file)
            n_mfcc=40
            obj=SpeechEmotionRecognition(file,n_mfcc)
            obj.ExtractFeatures()
            z=obj.mfcc
            ypr=model.predict([z])
            print(ypr)
        ans=ypr[0]


    return render_template("index.html",ans=ans)


if __name__=="__main__":
    app.run(debug=True)