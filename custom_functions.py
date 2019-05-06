import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import datetime

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (roc_auc_score,roc_curve,precision_recall_curve, auc,
                             classification_report, confusion_matrix, average_precision_score,
                             accuracy_score,silhouette_score,mean_squared_error)
from sklearn.utils.fixes import signature

def print_classification_performance2class_report(model,X_test,y_test):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    sns.set()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    PC =  TP/(TP+FP)
    RC = TP/(TP+FN)
    FS = 2 *((PC*RC)/(PC+RC))
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Accuracy:{:.2%}".format(ACC))
    print("Precision:{:.2%}".format(PC))
    print("Recall:{:.2%}".format(RC))
    print("Fscore:{:.2%}".format(FS))
    print("Average precision:{:.2%}".format(AP))
    print('The RMSE value is {:.4f}'.format(RMSE))
    
    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(142)
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini))
    plt.legend(loc='lower right')
    
    #pr
    plt.subplot(143)
    precision, recall, _ = precision_recall_curve(y_test,y_pred_proba)
    step_kwargs = ({'step':'post'}
                  if 'step'in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2, where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP))
    
    #hist
    plt.subplot(144)
    tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    plt.show()
    
    return ACC,PC,RC,FS,AP,roc_auc,gini,RMSE


def generate_miss_report(data):
    """ 
        Program: print missing report
        Author: Siraprapa W.
        
        Purpose: print missing report of a dataset
    """
    
    def set_nan(x):
        ch_miss = ['n/a','na','n.a.','n.a','*','-','unknown'
                   ,'email@domain.com','testuser','u','99999999'
                   ,'null','none','c999','z_error','z_missing']
        nm_miss = [99999999.0,-999.0]
        
        if type(x) is str :
            if x.lower() in ch_miss :
                return np.nan
            else:
                return x
        elif x in nm_miss:
             return np.nan
        else:
             return x
    
    data_dub = data.copy().applymap(set_nan)
    n_miss  = data_dub.isnull().sum() 
    n_miss = n_miss.reset_index()
    p_miss  = data_dub.isnull().sum() / len(data_dub.index)
    p_miss = p_miss.reset_index()
    miss_report = pd.merge(n_miss,p_miss,how='inner',on='index')
    miss_report.columns= ['feature','n_missing','p_missing']
    miss_report['n_populated'] = len(data_dub.index) - miss_report['n_missing']
    miss_report['p_populated'] = 1.0 - miss_report['p_missing']
    
    miss_report.plot(kind='bar',x='feature',y='p_populated',color='lightblue')
    
    print(data_dub.dtypes)
    
    return miss_report



def print_gainlift_charts(data,var_to_rank,var_to_count_nonzero,intpl=False,silent=False):
    """
        Program: print_gainlift_charts
        Author: Siraprapa W.
        
        Purpose:print gain lift charts   
    """
    n_qcut=10
    if intpl==False:
        tmpname = var_to_rank+'_tmp'
        data[tmpname] = pd.qcut(data[var_to_rank],n_qcut,labels=False)
        table = pd.pivot_table(data, 
                                 index=[tmpname],
                                 values=[var_to_rank,var_to_count_nonzero],
                                 aggfunc={
                                     var_to_rank:np.size,
                                     var_to_count_nonzero:np.count_nonzero
                                 }
                                )
        table_sorted = table.sort_index(ascending=False)
        table_sorted['cumulative_response'] = table_sorted[var_to_count_nonzero].cumsum()
        
        table_sorted['nonresponse'] = table_sorted[var_to_rank]-table_sorted[var_to_count_nonzero]
        table_sorted['cumulative_nonresponse'] = table_sorted['nonresponse'].cumsum()
        
        total_nonresponse = np.sum(table_sorted.loc[:,'nonresponse'])
        table_sorted['percent_of_nonevents'] = (table_sorted['nonresponse']/total_nonresponse)
        table_sorted['cumulative_percent_of_nonevents'] = table_sorted['percent_of_nonevents'].cumsum()
        
        total_response = np.sum(table_sorted.loc[:,var_to_count_nonzero])
        table_sorted['percent_of_events'] = (table_sorted[var_to_count_nonzero]/total_response)
        table_sorted['cumulative_percent_of_events'] = table_sorted['percent_of_events'].cumsum()
        
        table_sorted['cumulative_gain'] = (table_sorted['cumulative_response']/total_response)*100
        decile = np.linspace(1,10,10)
        table_sorted['decile'] = decile*10
        table_sorted['cumulative_lift'] = table_sorted['cumulative_gain']/(table_sorted['decile'])
        table_sorted.rename(columns={var_to_rank: 'counts'},inplace=True)
        table_sorted['ks'] = np.abs(table_sorted['cumulative_percent_of_events']-table_sorted['cumulative_percent_of_nonevents'])
        #table_sorted.set_index('decile')
        
    else:
        list_binsize= []
        list_counts = []
        inspace_qcut= np.linspace(np.min(data[var_to_rank]),np.max(data[var_to_rank]),n_qcut)
        for i in range(len(inspace_qcut)):
            if i == (len(inspace_qcut)-1):
                mask = (data[var_to_rank]>= (len(inspace_qcut)-1))
                tocal = data.loc[mask,[var_to_rank,var_to_count_nonzero]]
                size = np.size(tocal[var_to_rank])
                counts = np.count_nonzero(tocal[var_to_count_nonzero])
                list_binsize.append(size)
                list_counts.append(counts)  

            else:
                mask = (data[var_to_rank]>=inspace_qcut[i]) & ( data[var_to_rank]<inspace_qcut[i+1])
                tocal = data.loc[mask,[var_to_rank,var_to_count_nonzero]]
                size = np.size(tocal[var_to_rank])
                counts = np.count_nonzero(tocal[var_to_count_nonzero])
                list_binsize.append(size)
                list_counts.append(counts)
            
        table = pd.DataFrame([list_binsize,list_counts]).transpose()
        table.columns = [var_to_rank,var_to_count_nonzero]

        table_sorted = table.sort_index(ascending=False)
        table_sorted['cumulative_response'] = table_sorted[var_to_count_nonzero].cumsum()
        
        table_sorted['nonresponse'] = table_sorted[var_to_rank]-table_sorted[var_to_count_nonzero]
        table_sorted['cumulative_nonresponse'] = table_sorted['nonresponse'].cumsum()
        
        total_nonresponse = np.sum(table_sorted.loc[:,'nonresponse'])
        table_sorted['percent_of_nonevents'] = (table_sorted['nonresponse']/total_nonresponse)
        table_sorted['cumulative_percent_of_nonevents'] = table_sorted['percent_of_nonevents'].cumsum()
        
        total_response = np.sum(table_sorted.loc[:,var_to_count_nonzero])
        table_sorted['percent_of_events'] = (table_sorted[var_to_count_nonzero]/total_response)
        table_sorted['cumulative_percent_of_events'] = table_sorted['percent_of_events'].cumsum()
        
        table_sorted['cumulative_gain'] = (table_sorted['cumulative_response']/total_response)*100
        decile = np.linspace(1,10,10)
        table_sorted['decile'] = decile*10
        table_sorted['cumulative_lift'] = table_sorted['cumulative_gain']/(table_sorted['decile'])
        table_sorted.rename(columns={var_to_rank: 'counts'},inplace=True)
        table_sorted['ks'] = np.abs(table_sorted['cumulative_percent_of_events']-table_sorted['cumulative_percent_of_nonevents'])
    
    table_sorted['base_lift']=1
    table_sorted['base_gain']=[x for x in np.linspace(10,100,10)]
    
    #charts
    if silent==False:

        fig, axes = plt.subplots(nrows=1, ncols=2)
        table_sorted.plot(kind='line',x='decile',y=['cumulative_lift','base_lift'],style='o-',
                          ax=axes[0],figsize=(10,3),title='{} - Cumulative Lift Curve'.format(var_to_rank))
        table_sorted.plot(kind='line',x='decile',y=['cumulative_gain','base_gain'],style='o-',
                          ax=axes[1],figsize=(10,3),title='{} - Cumulative Gain Curve'.format(var_to_rank))


        print(table_sorted.set_index('decile').loc[:,[var_to_count_nonzero,'cumulative_gain','cumulative_lift','ks']])
    else:
        pass
    
    return table_sorted.set_index('decile').drop(columns=['base_lift','base_gain'],axis=1)
    
#