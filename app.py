from contextlib import AsyncExitStack
from flask import Flask,render_template,request,redirect

import pickle

model = pickle.load(open('model_pickle','rb'))
print(model)
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
        ans='33333'


    return render_template("index.html",ans=ans)


if __name__=="__main__":
    app.run(debug=True)