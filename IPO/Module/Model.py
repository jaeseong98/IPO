#!/usr/bin/env python
# coding: utf-8
# %%
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from dateutil.parser import parse
import copy

from imblearn.over_sampling import SMOTE
#시각화
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import *

##알고리즘 
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# %%
def Cut(Series , cuts):
    "구분할 iter가능한 변수와 구분 기준을 입력받으면 구분 기준 앞에서 부터 1로 구분해서 return 함"
    Cuts = copy.deepcopy(cuts)
    Cuts.append(np.inf)
    Cuts.insert(0,-1*np.inf)
    R = len(Cuts)
    label = range(1,R)
    return pd.cut(Series, Cuts, labels = label)
# ## 변수 위치 재설정


# %% [markdown]
# # 평가지표

# %%
def get_clf_eval(y_test, y_data, pred=None, pred_proba=None): 
    confusion = confusion_matrix(y_test, pred) 
    accuracy = accuracy_score(y_test, pred) 
    precision = precision_val(y_data)
    recall = recall_val(y_data) 
    f1 = f1_score(y_test, pred , average= 'weighted') 
    
#     print('정확도(accuracy): {0:.4f}, 정밀도(precision): {1:.4f}, 재현율(recall): {2:.4f}, f1_score: {3:.4f}'.format(accuracy, precision, recall, f1))
    

    return [accuracy, precision, recall, f1]


# %%
def precision_val(y_data):
    """
    본 연구에서 새로 구현한 정밀도 계산 식
    """
    new = y_data[y_data['예측 y 라벨링'] > 2]
    real = new[new['실제 y라벨링'] > 2]
    if len(new) == 0:
        per = 0
    else :
        per = len(real)/len(new)
    return per


# %%
def recall_val(y_data):
    """
    본 연구에서 새로 구현한 재현율 계산 식
    """
    real = y_data[y_data['실제 y라벨링'] > 2]
    new = real[real['예측 y 라벨링'] > 2]
    if len(real) == 0:
        per = 0
    else :
        per = len(new)/len(real)
    return per


# %% [markdown]
# ## 수익률 계산

# %%
def buy_sell(y_data):
    """
    buy는 구매하고 sell은 팔았을때 수익률 값
    """
    portfolio = []
    for i in range(len(y_data)):
        buy = y_data[i][y_data[i]['예측 y 라벨링'] > 2]['실제 y 수익률']
        sell = y_data[i][y_data[i]['예측 y 라벨링'] < 2]['실제 y 수익률']
        sum_1 = buy.sum() - sell.sum()
        sum_2 = len(buy)+len(sell)
        value = sum_1/sum_2
        portfolio.append(value)
    data = pd.DataFrame(portfolio).T
    data.rename(index = {0 : "buy_sell수익률"},columns = lambda x : "model_set_"+ str(x),inplace = True )
    data['mean'] = data.mean(axis=1)[0]
                         
    return data
    


# %%
def buy(y_data):
    """
    buy만 구매했을때 수익률
    """
    portfolio = []
    for i in range(len(y_data)):
        buy = y_data[i][y_data[i]['예측 y 라벨링'] > 2]['실제 y 수익률']
        sum_1 = buy.sum()
        sum_2 = len(buy)
        value = sum_1/sum_2
        portfolio.append(value)
    data = pd.DataFrame(portfolio).T
    data.rename(index = {0 : "buy수익률"},columns = lambda x : "model_set_"+ str(x),inplace = True )
    data['mean'] = data.mean(axis=1)[0]
                         
    return data


# %%
def mean(y_data):
    """
    전부 구매했을때 수익률
    """
    portfolio = []
    for i in range(len(y_data)):
        buy = y_data[i]['실제 y 수익률']
        sum_1 = buy.sum()
        sum_2 = len(buy)
        value = sum_1/sum_2
        portfolio.append(value)
    data = pd.DataFrame(portfolio).T
    data.rename(index = {0 : "all수익률"},columns = lambda x : "model_set_"+ str(x),inplace = True )
    data['mean'] = data.mean(axis=1)[0]
    return data


# %%
def frame(y_data):
    """
    위의 3가지 구분별 수익률 값을 데이터 프레임으로 변환
    """
    buy_frame = buy(y_data)
    buy_sell_frame = buy_sell(y_data)
    mean_frame = mean(y_data)
    value = pd.concat([buy_sell_frame,buy_frame,mean_frame])
    return value


# %% [markdown]
# # 랜덤포레스트
# ## 공통 전처리
# 1. 종목명과 공모 시가총액 변수 Drop
# 2. 상장일을 인덱스로 설정 후 종속변수 별로 카테고리 분류
# 3. train 4년 test1년으로 총 기간을 3개월 이동하며 32개 모델 분할
# 4. 데이터 분포가 매우 불균형하고 적기에 오버샘플링(SMOTE) 진행
# 5. 랜덤포레스트,XGB,LGBM의 하이퍼파라미터들을 수정하며 최적 파라미터에 대해 모델 분석
# 6. 32개 모델에 대해 평가성과 , 변수중요도 , 예측결과 및 실제Y분포 출력

# %%
def process_final(df,y_name):

    ## 기본 전처리
    df['상장일'] = pd.to_datetime(df['상장일'])
    df = df.set_index(['상장일'])
    df = df.drop(['종목명','공모 시가총액'],axis = 1) ## 나중에 카테고리 진행할려면 남겨줄 것
    if y_name == '공모가 대비 6개월 수익률' or "종가 대비 6개월 수익률":
        df['Cat'] = Cut(df[y_name],[ -0.2, 0.2, 0.4])
       
        
    else :
        df['Cat'] = Cut(df[y_name],[ -0.1, 0.1, 0.2])
       
    
    ## train 4년 test 1년으로 총 기간을 3개월 이동으로 32개 구간분할
    train_list = []
    test_list = []
    train_start_date = '2009-04-01' ## 기한은 나중에 변경할수도
    test_start_date = parse(str(train_start_date)).date() + relativedelta(years =4)
    train_end_date = parse(str(test_start_date)).date() - relativedelta(days =1)
    test_end_date = parse(str(train_end_date)).date() + relativedelta(years =1) 

    while True:

        train_list.append(df[train_start_date : train_end_date])
        test_list.append(df[test_start_date : test_end_date])

        train_start_date = parse(str(train_start_date)).date() + relativedelta(months=3)
        test_start_date = parse(str(test_start_date)).date() + relativedelta(months=3)
        train_end_date = parse(str(train_end_date)).date() + relativedelta(months=3)
        test_end_date = parse(str(test_end_date)).date() + relativedelta(months=3)

        if str(train_start_date) == '2017-04-01':
            break
            
    y_data = []
    score_list = []
    feature_list = []

    for i in range(0,len(train_list)):
        X_train_before = train_list[i].drop([y_name,'Cat'],axis =1)
        y_train_before = train_list[i]['Cat']
        X_test = test_list[i].drop([y_name,'Cat'],axis =1)
        y_test = test_list[i]['Cat']
        
        smote = SMOTE(random_state=0,k_neighbors = 3)
        X_train,y_train = smote.fit_resample(X_train_before,y_train_before)
#         ros = RandomOverSampler()
#         X_train,y_train = ros.fit_resample(X_train_before,y_train_before)

       # 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
              
        clf = RandomForestClassifier(random_state=0 , max_depth = 4 , min_samples_leaf = 6 , min_samples_split =  12 , \
                                     n_estimators = 20 )
        

        clf.fit(X_train , y_train)
        train_pred = clf.predict(X_train_before)
        test_pred = clf.predict(X_test)




        #feature_importance
        feature_importance = clf.feature_importances_


        pred_value = pd.Series(test_pred,index = y_test.index)
        per = test_list[i][y_name]
        y_testdata = pd.concat([per,y_test,pred_value] , axis = 1)
        y_testdata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        train_pred_value = pd.Series(train_pred,index = y_train_before.index)
        per = train_list[i][y_name]
        y_traindata = pd.concat([per,y_train_before,train_pred_value] , axis = 1)
        y_traindata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        
        # 성과 평가
        train_res = get_clf_eval(y_train_before, y_traindata, train_pred)
        test_res = get_clf_eval(y_test, y_testdata,test_pred)
        res = train_res + test_res
        
        score_list.append(res)
        feature_list.append(feature_importance)
        y_data.append(y_testdata)   
    df_score = pd.DataFrame(score_list,columns = ["정확도","정밀도","재현율","f1_score"]+["정확도_test","정밀도_test","재현율_test","f1_score_test"]).T 
    df_score.rename(columns = lambda x : "model_set_"+ str(x),inplace = True)

    df_feature = pd.DataFrame(feature_list,columns = X_train.columns).T 
    df_feature.rename(columns = lambda x : "model_set_"+ str(x),inplace = True)
        
    return df_score , df_feature , y_data


# %% [markdown]
# ## XGBoost

# %%
def process_xgb(df,y_name):
    

    ## 기본 전처리
    df['상장일'] = pd.to_datetime(df['상장일'])
    df = df.set_index(['상장일'])
    df = df.drop(['종목명','공모 시가총액'],axis = 1) ## 나중에 카테고리 진행할려면 남겨줄 것
    if y_name == '공모가 대비 6개월 수익률' or "종가 대비 6개월 수익률":
        df['Cat'] = Cut(df[y_name],[ -0.2, 0.2, 0.4])
       
        
    else :
        df['Cat'] = Cut(df[y_name],[ -0.1, 0.1, 0.2])
       
    
    ## train 3년 test 1년으로 총 기간을 3개월 이동으로 36개 구간분할
    train_list = []
    test_list = []
    train_start_date = '2009-04-01' ## 기한은 나중에 변경할수도
    test_start_date = parse(str(train_start_date)).date() + relativedelta(years =4)
    train_end_date = parse(str(test_start_date)).date() - relativedelta(days =1)
    test_end_date = parse(str(train_end_date)).date() + relativedelta(years =1) 

    while True:

        train_list.append(df[train_start_date : train_end_date])
        test_list.append(df[test_start_date : test_end_date])

        train_start_date = parse(str(train_start_date)).date() + relativedelta(months=3)
        test_start_date = parse(str(test_start_date)).date() + relativedelta(months=3)
        train_end_date = parse(str(train_end_date)).date() + relativedelta(months=3)
        test_end_date = parse(str(test_end_date)).date() + relativedelta(months=3)

        if str(train_start_date) == '2017-04-01':
            break
            
    y_data = []
    score_list = []
    feature_list = []

    for i in range(0,len(train_list)):
        X_train_before = train_list[i].drop([y_name,'Cat'],axis =1)
        y_train_before = train_list[i]['Cat']
        X_test = test_list[i].drop([y_name,'Cat'],axis =1)
        y_test = test_list[i]['Cat']
        
        smote = SMOTE(random_state=0,k_neighbors = 3)
        X_train,y_train = smote.fit_resample(X_train_before,y_train_before)
#         ros = RandomOverSampler()
#         X_train,y_train = ros.fit_resample(X_train_before,y_train_before)

       # 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
              
        clf =  XGBClassifier(random_state = 0 ,colsample_bytree = 1, gamma =  0.01,\
                            learning_rate =  0.01, max_depth =  4, reg_lambda =  0, subsample =  1,
                             eval_metric = "mlogloss")
        

        clf.fit(X_train , y_train)
        train_pred = clf.predict(X_train_before)
        test_pred = clf.predict(X_test)




        #feature_importance
        feature_importance = clf.feature_importances_


        pred_value = pd.Series(test_pred,index = y_test.index)
        per = test_list[i][y_name]
        y_testdata = pd.concat([per,y_test,pred_value] , axis = 1)
        y_testdata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        train_pred_value = pd.Series(train_pred,index = y_train_before.index)
        per = train_list[i][y_name]
        y_traindata = pd.concat([per,y_train_before,train_pred_value] , axis = 1)
        y_traindata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        
        # 성과 평가
        train_res = get_clf_eval(y_train_before, y_traindata, train_pred)
        test_res = get_clf_eval(y_test, y_testdata,test_pred)
        res = train_res + test_res
        
        score_list.append(res)
    df_score = pd.DataFrame(score_list,columns = ["정확도","정밀도","재현율","f1_score"]+["정확도_test","정밀도_test","재현율_test","f1_score_test"]).T 
    df_score.rename(columns = lambda x : "model_set_"+ str(x),inplace = True)


    return df_score 


# %% [markdown]
# ## lgbm

# %%
def process_lgbm(df,y_name):
    


    
    ## 기본 전처리
    df['상장일'] = pd.to_datetime(df['상장일'])
    df = df.set_index(['상장일'])
    df = df.drop(['종목명','공모 시가총액'],axis = 1) ## 나중에 카테고리 진행할려면 남겨줄 것
    if y_name == '공모가 대비 6개월 수익률' or "종가 대비 6개월 수익률":
        df['Cat'] = Cut(df[y_name],[ -0.2, 0.2, 0.4])
       
        
    else :
        df['Cat'] = Cut(df[y_name],[ -0.1, 0.1, 0.2])
       
    
    ## train 3년 test 1년으로 총 기간을 3개월 이동으로 36개 구간분할
    train_list = []
    test_list = []
    train_start_date = '2009-04-01' ## 기한은 나중에 변경할수도
    test_start_date = parse(str(train_start_date)).date() + relativedelta(years =4)
    train_end_date = parse(str(test_start_date)).date() - relativedelta(days =1)
    test_end_date = parse(str(train_end_date)).date() + relativedelta(years =1) 

    while True:

        train_list.append(df[train_start_date : train_end_date])
        test_list.append(df[test_start_date : test_end_date])

        train_start_date = parse(str(train_start_date)).date() + relativedelta(months=3)
        test_start_date = parse(str(test_start_date)).date() + relativedelta(months=3)
        train_end_date = parse(str(train_end_date)).date() + relativedelta(months=3)
        test_end_date = parse(str(test_end_date)).date() + relativedelta(months=3)

        if str(train_start_date) == '2017-04-01':
            break
            
    y_data = []
    score_list = []
    feature_list = []

    for i in range(0,len(train_list)):
        X_train_before = train_list[i].drop([y_name,'Cat'],axis =1)
        y_train_before = train_list[i]['Cat']
        X_test = test_list[i].drop([y_name,'Cat'],axis =1)
        y_test = test_list[i]['Cat']
        
        smote = SMOTE(random_state=0,k_neighbors = 3)
        X_train,y_train = smote.fit_resample(X_train_before,y_train_before)
#         ros = RandomOverSampler()
#         X_train,y_train = ros.fit_resample(X_train_before,y_train_before)

       # 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
              
        clf =  LGBMClassifier(random_state =0, min_data_in_leaf = 20 , reg_alpha = 0.1,n_estimators=30,max_depth = 5)


        clf.fit(X_train , y_train)
        train_pred = clf.predict(X_train_before)
        test_pred = clf.predict(X_test)




        #feature_importance
        feature_importance = clf.feature_importances_


        pred_value = pd.Series(test_pred,index = y_test.index)
        per = test_list[i][y_name]
        y_testdata = pd.concat([per,y_test,pred_value] , axis = 1)
        y_testdata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        train_pred_value = pd.Series(train_pred,index = y_train_before.index)
        per = train_list[i][y_name]
        y_traindata = pd.concat([per,y_train_before,train_pred_value] , axis = 1)
        y_traindata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        
        # 성과 평가
        train_res = get_clf_eval(y_train_before, y_traindata, train_pred)
        test_res = get_clf_eval(y_test, y_testdata,test_pred)
        res = train_res + test_res
        
        score_list.append(res)
    df_score = pd.DataFrame(score_list,columns = ["정확도","정밀도","재현율","f1_score"]+["정확도_test","정밀도_test","재현율_test","f1_score_test"]).T 
    df_score.rename(columns = lambda x : "model_set_"+ str(x),inplace = True)


        
    return df_score


# %% [markdown]
# ## ROS
# > 오버샘플링 기법중 SMOTE와 ROS 비교를 위한 ROS 코드

# %%
def process_final_ros(df,y_name):
    
    ## 기본 전처리
    df['상장일'] = pd.to_datetime(df['상장일'])
    df = df.set_index(['상장일'])
    df = df.drop(['종목명','공모 시가총액'],axis = 1) ## 나중에 카테고리 진행할려면 남겨줄 것
    if y_name == '공모가 대비 6개월 수익률' or "종가 대비 6개월 수익률":
        df['Cat'] = Cut(df[y_name],[ -0.2, 0.2, 0.4])
       
        
    else :
        df['Cat'] = Cut(df[y_name],[ -0.1, 0.1, 0.2])
       
    
    ## train 3년 test 1년으로 총 기간을 3개월 이동으로 36개 구간분할
    train_list = []
    test_list = []
    train_start_date = '2009-04-01' ## 기한은 나중에 변경할수도
    test_start_date = parse(str(train_start_date)).date() + relativedelta(years =4)
    train_end_date = parse(str(test_start_date)).date() - relativedelta(days =1)
    test_end_date = parse(str(train_end_date)).date() + relativedelta(years =1) 

    while True:

        train_list.append(df[train_start_date : train_end_date])
        test_list.append(df[test_start_date : test_end_date])

        train_start_date = parse(str(train_start_date)).date() + relativedelta(months=3)
        test_start_date = parse(str(test_start_date)).date() + relativedelta(months=3)
        train_end_date = parse(str(train_end_date)).date() + relativedelta(months=3)
        test_end_date = parse(str(test_end_date)).date() + relativedelta(months=3)

        if str(train_start_date) == '2017-04-01':
            break
            
    y_data = []
    score_list = []
    feature_list = []

    for i in range(0,len(train_list)):
        X_train_before = train_list[i].drop([y_name,'Cat'],axis =1)
        y_train_before = train_list[i]['Cat']
        X_test = test_list[i].drop([y_name,'Cat'],axis =1)
        y_test = test_list[i]['Cat']
        
#         smote = SMOTE(random_state=0,k_neighbors = 3)
#         X_train,y_train = smote.fit_resample(X_train_before,y_train_before)
        ros = RandomOverSampler()
        X_train,y_train = ros.fit_resample(X_train_before,y_train_before)

       # 랜덤 포레스트 학습 및 별도의 테스트 셋으로 예측 성능 평가
              
        clf = RandomForestClassifier(random_state=0 , max_depth = 4 , min_samples_leaf = 6 , min_samples_split =  12 , \
                                     n_estimators = 20 )
        

        clf.fit(X_train , y_train)
        train_pred = clf.predict(X_train_before)
        test_pred = clf.predict(X_test)




        #feature_importance
        feature_importance = clf.feature_importances_


        pred_value = pd.Series(test_pred,index = y_test.index)
        per = test_list[i][y_name]
        y_testdata = pd.concat([per,y_test,pred_value] , axis = 1)
        y_testdata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        train_pred_value = pd.Series(train_pred,index = y_train_before.index)
        per = train_list[i][y_name]
        y_traindata = pd.concat([per,y_train_before,train_pred_value] , axis = 1)
        y_traindata.columns = ['실제 y 수익률','실제 y라벨링','예측 y 라벨링']
        
        
        # 성과 평가
        train_res = get_clf_eval(y_train_before, y_traindata, train_pred)
        test_res = get_clf_eval(y_test, y_testdata,test_pred)
        res = train_res + test_res
        
        score_list.append(res)
        feature_list.append(feature_importance)
        y_data.append(y_testdata)   
    df_score = pd.DataFrame(score_list,columns = ["정확도","정밀도","재현율","f1_score"]+["정확도_test","정밀도_test","재현율_test","f1_score_test"]).T 
    df_score.rename(columns = lambda x : "model_set_"+ str(x),inplace = True)


        
    return df_score 
