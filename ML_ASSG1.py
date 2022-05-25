from sklearn import datasets
from sklearn import tree
#evaluation
from sklearn import metrics
#cross validation
from sklearn import model_selection
#Random forest
from sklearn.ensemble import RandomForestClassifier
#Bagging
from sklearn.ensemble import BaggingClassifier
#Adaboost
from sklearn.ensemble import AdaBoostClassifier

data = datasets.fetch_openml(data_id = 1471)
features = data.feature_names
data_val = data.data
dat_target = data.target
#type(data.feature_names)
#type(data.data)
#type(data.target)

mytree_entropy = tree.DecisionTreeClassifier(criterion="entropy")
mytree_gini = tree.DecisionTreeClassifier(criterion="gini")
tree_entropy = mytree_entropy.fit(data.data,dat_target)
tree_gini = mytree_gini.fit(data_val,dat_target)
print("entropy Tree",tree.export_text(mytree_entropy))
print("GINI TREE",tree.export_text(mytree_gini))
predictions = mytree_entropy.predict(data_val)
predictions_gini = mytree_gini.predict(data_val)
#type(predictions)
print("predictions",predictions)
print("predictions",predictions_gini)
#Evaluation
print("Evaluation using criterion as entropy")
a=metrics.accuracy_score(dat_target, predictions)
f1=metrics.f1_score(dat_target, predictions, pos_label='1')
p=metrics.precision_score(dat_target, predictions, pos_label="1")
r=metrics.recall_score(dat_target, predictions, pos_label='1')
print("accuracy:",a,"F1-measure:",f1,"Precision:",p,"recall:",r)
print("Evaluation using criterion as GINI")
a_g=metrics.accuracy_score(dat_target, predictions_gini)
f1_g=metrics.f1_score(dat_target, predictions_gini, pos_label='1')
p_g=metrics.precision_score(dat_target, predictions_gini, pos_label="1")
r_g=metrics.recall_score(dat_target, predictions_gini, pos_label='1')
print("accuracy:",a_g,"F1-measure:",f1_g,"Precision:",p_g,"recall:",r_g)

#crossvalidation

dtc = tree.DecisionTreeClassifier()
dtc_entropy = tree.DecisionTreeClassifier(criterion="entropy")
cv = model_selection.cross_validate(dtc, data_val, dat_target, scoring="roc_auc", cv=9)
#type(cv)
#print("Cross validation with roc_auc",cv)
cv['test_score']
m_roc=cv['test_score'].mean()
print("mean for test_score for roc_auc",m_roc)
#cv_acc_roc = model_selection.cross_validate(dtc, data_val, dat_target, scoring=["accuracy","roc_auc"], cv=9)
#print("Cross validation with accuracy and roc_auc",cv_acc_roc)
#m_acc=cv_acc_roc['test_accuracy'].mean()
#m_r= cv_acc_roc['test_roc_auc'].mean()
#print("mean for test_score for roc_auc",m_r,"mean for test accuracy",m_acc)
#cv_w_train = model_selection.cross_validate(dtc, data_val, dat_target, scoring=["accuracy","roc_auc"], cv=9, return_train_score=True)
#print("with training scores",cv_w_train)
#cv_w_train["train_accuracy"].mean()
#cv_w_train["train_roc_auc"].mean()

#entropy
print("tree Cross validation on entropy")
cv_entropy = model_selection.cross_validate(dtc_entropy, data.data, data.target, scoring=["roc_auc"], cv=9)
print("tree cv with entropy",cv_entropy)
m_t_ent= cv_entropy['test_roc_auc'].mean()
#m_t_ent2= cv_entropy["test_accuracy"].mean()
print("mean testscore of entropy",m_t_ent)
#print("mean testscore of entropy",m_t_ent2)
#parameter tunning
parameters = [{"min_samples_leaf":[10,20,30,40,50]}]
print("parameters",parameters)
#GINI
print("tree Cross validation on GINI")
dtc = tree.DecisionTreeClassifier()
tuned_dtc = model_selection.GridSearchCV(dtc, parameters, scoring="roc_auc", cv=10)
cv1 = model_selection.cross_validate(tuned_dtc, data.data, data.target, scoring=["roc_auc"], cv=10, return_train_score=True)
print("tunned tree cv with gini",cv1)
m_t_gc1 = cv1['test_roc_auc'].mean()
#m_t_gc2 = cv1["test_accuracy"].mean()
print("tunned tree mean testscore of gini",m_t_gc1)
#print("tunned tree mean testscore of gini",m_t_gc2)
tuned_dtc.fit(data.data, data.target)
param = tuned_dtc.best_params_
print("tuned on:",param)
#entropy

print("tree Cross validation on entropy")
dtc_entropy = tree.DecisionTreeClassifier(criterion="entropy")
tuned_dtc_ent = model_selection.GridSearchCV(dtc_entropy, parameters, scoring="roc_auc", cv=10)
cv_entropy = model_selection.cross_validate(dtc_entropy, data.data, data.target, scoring=["roc_auc"], cv=9)
print("tunned tree cv with entropy",cv_entropy)
m_t_ent1= cv1['test_roc_auc'].mean()
#m_t_ent2= cv1["test_accuracy"].mean()
print("tunned tree mean testscore of entropy",m_t_ent1)
#print("tunned tree mean testscore of entropy",m_t_ent2)
tuned_dtc_ent.fit(data.data, data.target)
param_e = tuned_dtc_ent.best_params_
print("tuned on:",param_e)

#Random Forest

print("Random Tree Cross validation on GINI")
rf = RandomForestClassifier()
cv_rf_g = model_selection.cross_validate(rf, data.data, data.target, scoring=["roc_auc"], cv=9)
print("Random tree CV with gini",cv_rf_g)
m_rf_g = cv_rf_g['test_roc_auc'].mean()
print("Random tree mean testscore of entropy",m_rf_g)

#Bagging

print("Bagging Cross validation on GINI")
bagged_dtc = BaggingClassifier()
cv_bg_g = model_selection.cross_validate(bagged_dtc, data.data, data.target, scoring=["roc_auc"], cv=9)
print("Bagging tree CV with gini",cv_bg_g)
m_bg_g = cv_bg_g['test_roc_auc'].mean()
print("Bagging mean testscore of gini",m_bg_g)

#entropy

#print("Bagging Cross validation on entropy")
#bagged_dtc_e = BaggingClassifier(criterion="entropy")
#cv_bg_e = model_selection.cross_validate(bagged_dtc_e, data.data, data.target, scoring="accuracy", cv=10)
#print("Bagging tree CV with entropy",cv_bg_e)
#m_bg_e = cv_rf['test_score'].mean()
#print("mean testscore of entropy",m_bg_e)


#AdaBoost

print("AdaBoost Cross validation on GINI")
ada_dtc = AdaBoostClassifier()
cv_ada_g = model_selection.cross_validate(ada_dtc,data.data,data.target,scoring =["roc_auc"],cv =9)
print("Adaboost tree CV with entropy",cv_ada_g)
m_ada_g = cv_ada_g['test_roc_auc'].mean()
print("Adaboost mean testscore of gini",m_ada_g)


#entropy


#print("AdaBoost Cross validation on GINI")
#ada_dtc_e = AdaBoostClassifier(criterion="entropy")
#cv_ada_e = model_selection.cross_validate(ada_dtc_e,data.data,data.target,scoring ="accuracy",cv =10)
#print("Adaboost tree CV with entropy",cv_ada_e)
#m_ada_e = cv_rf['test_score'].mean()
#print("mean testscore of gini",m_ada_e)