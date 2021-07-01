from django.shortcuts import render
import pickle
import numpy as np
def home(request):
    return render(request, 'index.html')
def get_prediction(pclass,sex,age,sibsp,parch,fare,C,Q,S):
    model=pickle.load(open("ml_model.sav",'rb'))
    a=np.asarray([pclass,sex,age,sibsp,parch,fare,C,Q,S])
    a=a.reshape(1,-1)
    prediction=model.predict(a)
    if prediction==0:
        return "no"
    elif prediction==1:
        return "yes"
    else:
        return 'error'
def result(request):
    pclass=int(request.GET['pclass'])
    sex=int(request.GET['sex'])
    age=int(request.GET['age'])
    sibsp=int(request.GET['sibsp'])
    parch=int(request.GET['parch'])
    fare=int(request.GET['fare'])
    embC=int(request.GET['embC'])
    embQ=int(request.GET['embQ'])
    embS=int(request.GET['embS'])
    result=get_prediction(pclass,sex,age,sibsp,parch,fare,embC,embQ,embS)
    return render(request,'result.html',{'result':result})
