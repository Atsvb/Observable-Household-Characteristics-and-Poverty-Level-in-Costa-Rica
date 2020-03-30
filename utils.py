import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2

def fold(data_set, indexes, n_splits=5):
    ##inputs:
    ##data_set: dataframe of households and their features
    ##indexes: household indexes
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_index, valid_index in kf.split(indexes):
        train_id=indexes[train_index]
        train_sp=data_set[data_set['homeid'].isin(train_id)]
        valid_id=indexes[valid_index]
        valid_sp=data_set[data_set['homeid'].isin(valid_id)]
        yield train_sp, valid_sp
        
def cross_validation(model, data_set, targets, indexes, importances=False, n_splits=5):
    ##inputs:
    ##model: classification model to be used 
    ##dataset: dataframe of houselholds and their features
    ##targets: number of classes
    ##indexes: household indexes
    ##importance: if true, output the feature importances
    rep=np.zeros([n_splits,3*targets+2])
    if importances:
        imp_array=np.zeros([n_splits, len(data_set.columns)-3])
    i=0
    for train_fold,valid_fold in fold(data_set, indexes, n_splits):
        X_train=train_fold.drop(['Target','Id', 'homeid'], axis=1)
        Y_train=train_fold.Target
        X_valid=valid_fold.drop(['Target','Id', 'homeid'], axis=1)
        Y_valid=valid_fold.Target
        model_fit=model.fit(X_train,Y_train)
        preds=model_fit.predict(X_valid)
        for k in range(targets):
            rep[i,0+3*k]=classification_report(Y_valid, preds, output_dict=True)[str(k+1)] ['precision' ]
            rep[i,1+3*k]=classification_report(Y_valid, preds, output_dict=True)[str(k+1)] ['recall' ]
            rep[i,2+3*k]=classification_report(Y_valid, preds, output_dict=True)[str(k+1)] ['f1-score' ]
        rep[i,-2]=classification_report(Y_valid, preds, output_dict=True)['macro avg'] ['f1-score' ]
        rep[i,-1]=classification_report(Y_valid, preds, output_dict=True)['accuracy']         
        if importances:
            imp_array[i]=model.feature_importances_
        i=i+1
    if importances:
        importances_list = pd.DataFrame({'Features': X_train.columns, 
                             'Importances': np.mean(imp_array, axis=0)})
        importances_list.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

        return np.mean(rep, axis=0), importances_list
    else:
        return np.mean(rep, axis=0)

def print_report(report):
    for i in range(len(report)//3):
        print("Class "+str(i+1)+ ', precision: '+str(report[3*i]))
        print("Class "+str(i+1)+ ', recall: '+str(report[3*i+1]))
        print("Class "+str(i+1)+ ', F1: '+str(report[3*i+2]))
    print("Macro Avg F1: " +str(report[-2]))
    print("Acc :" + str(report[-1]))
    
def feature_selection(dataset, indexes, score_function=f_classif, n_features=60):
    fs = SelectKBest(score_function, n_features)
    dataset_sub=dataset[dataset['homeid'].isin(indexes)]
    X_train=dataset_sub.drop(['Target','Id', 'homeid'], axis=1)
    Y_train=dataset_sub.Target
    feature_sel = fs.fit(X_train, Y_train)
    selected=X_train.columns[feature_sel.get_support()]
    a=dataset.columns.get_loc('Target')
    b=dataset.columns.get_loc('Id')
    c=dataset.columns.get_loc('homeid')
    selected= selected.append([dataset_sub.columns[a:a+1],
                                               dataset_sub.columns[b:b+1],
                                               dataset_sub.columns[c:c+1]])
    return selected