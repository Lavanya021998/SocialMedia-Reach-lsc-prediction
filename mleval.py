
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def regval(xtrain, xtrain_poly, xtest, xtest_poly, ytrain, ytest, models):
    
    trainrmse = []
    testrmse = []
    trainr2 = []
    testr2 = []

    fit = []
    crossvalscore = []
    
    for name, model in models.items():
    
        if name!='poly':
            # RMSE, R2score
            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            trainrmse.append(round(np.sqrt(mean_squared_error(ytrain, ytrain_pred)),2))
            testrmse.append(round(np.sqrt(mean_squared_error(ytest, ytest_pred)),2))
            trainr2.append(round(r2_score(ytrain, ytrain_pred),2))
            testr2.append(round(r2_score(ytest, ytest_pred),2))
            trscore = r2_score(ytrain, ytrain_pred)
            tescore = r2_score(ytest, ytest_pred)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            # Cross-val score
            crossX = pd.concat([xtrain, xtest], axis = 0)
            crossy = pd.concat([ytrain, ytest], axis = 0)

            scores = cross_val_score(models[name], crossX, crossy, cv=3) # Taking 3 folds
            crossvalscore.append(round(scores.mean(),2))
        else:
            # RMSE, R2score
            ytrain_pred = model.predict(xtrain_poly)
            ytest_pred = model.predict(xtest_poly)

            trainrmse.append(round(np.sqrt(mean_squared_error(ytrain, ytrain_pred)),2))
            testrmse.append(round(np.sqrt(mean_squared_error(ytest, ytest_pred)),2))
            trainr2.append(round(r2_score(ytrain, ytrain_pred),2))
            testr2.append(round(r2_score(ytest, ytest_pred),2))
            trscore = r2_score(ytrain, ytrain_pred)
            tescore = r2_score(ytest, ytest_pred)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            # Cross-val score
            crossX = pd.concat([pd.DataFrame(xtrain_poly), pd.DataFrame(xtest_poly)], axis = 0)
            crossy = pd.concat([ytrain, ytest], axis = 0)

            scores = cross_val_score(models[name], crossX, crossy, cv=3) # Taking 3 folds
            crossvalscore.append(round(scores.mean(),2))
        
    return trainrmse, testrmse, trainr2, testr2, crossvalscore, fit



def classval(X, y, xtrain, xtest, ytrain, ytest, models):
    
    trainscore = []
    testscore = []
    fit = []
    crossvalscore = []
    
    for name, model in models.items():

        if name == 'xgb':

            
            ytrain_xg = np.where(ytrain == 'low', 0, np.where(ytrain == 'moderate', 1, 2))
            ytest_xg = np.where(ytest == 'low', 0, np.where(ytest == 'moderate', 1, 2))

            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            # Accuracy Score
            trscore = round(accuracy_score(ytrain_xg, ytrain_pred),2)
            tescore = round(accuracy_score(ytest_xg, ytest_pred),2)

            trainscore.append(trscore)
            testscore.append(tescore)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            y_xg = np.where(y== 'low', 0, np.where(y == 'moderate', 1, 2))

            # Cross-val score
            scores = cross_val_score(model, X, y_xg, cv=3,scoring='f1_micro')
            crossvalscore.append(round(scores.mean(),2))

        else:

            ytrain_pred = model.predict(xtrain.values)
            ytest_pred = model.predict(xtest.values)

            # Accuracy Score
            trscore = round(accuracy_score(ytrain, ytrain_pred),2)
            tescore = round(accuracy_score(ytest, ytest_pred),2)

            trainscore.append(trscore)
            testscore.append(tescore)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            # Cross-val score
            scores = cross_val_score(model, X.values, y, cv=3,scoring='f1_micro')
            crossvalscore.append(round(scores.mean(),2))  
            
    return trainscore, testscore, crossvalscore, fit


def imbclassval(X, y, xtrain, xtest, ytrain, ytest, models):
    
    trainscore = []
    testscore = []
    fit = []
    crossvalscore = []
    
    for name, model in models.items():

        if name == 'xgb':

            ytrain_xg = le.transform(ytrain)
            ytest_xg = le.transform(ytest)

            ytrain_pred = model.predict(xtrain)
            ytest_pred = model.predict(xtest)

            # f1 Score
            trscore = round(f1_score(ytrain_xg, ytrain_pred, average='micro'),2)
            tescore = round(f1_score(ytest_xg, ytest_pred, average='micro'),2)

            trainscore.append(trscore)
            testscore.append(tescore)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            y_xg = le.transform(y)

            # Cross-val score
            scores = cross_val_score(model, X, y_xg, cv=3, scoring='f1_micro')
            crossvalscore.append(round(scores.mean(),2))

        else:

            ytrain_pred = model.predict(xtrain.values)
            ytest_pred = model.predict(xtest.values)

            # f1 Score
            trscore = round(f1_score(ytrain, ytrain_pred, average='micro'),2)
            tescore = round(f1_score(ytest, ytest_pred, average='micro'),2)

            trainscore.append(trscore)
            testscore.append(tescore)

            # Bias-Variance Trade off
            if trscore<0.50 and tescore<0.50:
                if abs(trscore)==0 and abs(tescore)==0:
                    fit.append("Nofit")
                else:
                    fit.append("Underfit")     

            else:
                if abs(trscore-tescore)<0.10:
                    fit.append("Goodfit")
                elif abs(trscore-tescore)>=0.10:
                    fit.append("Overfit")
                else:
                    fit.append("Fit")

            # Cross-val score
            scores = cross_val_score(model, X.values, y, cv=3, scoring='f1_micro')
            crossvalscore.append(round(scores.mean(),2))  
            
    return trainscore, testscore, crossvalscore, fit