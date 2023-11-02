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
                             accuracy_score,silhouette_score,mean_squared_error,recall_score,precision_score,f1_score)
#from sklearn.utils.fixes import signature
import matplotlib.gridspec as gridspec
from inspect import signature

def print_classification_performance2class_report(model,X_test,y_test,is_lgb=False):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    sns.set()
    if is_lgb:
        y_pred_proba = model.predict(X_test)
        y_pred =  [1 if prob >= 0.5 else 0 for prob in y_pred_proba]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    
    #print(conf_mat)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    PC =  precision_score(y_test,y_pred)
    RC = recall_score(y_test,y_pred)
    FS = f1_score(y_test,y_pred)
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    
    print("Accuracy:{:.2%}".format(ACC)," | Precision:{:.2%}".format(PC)," | Recall:{:.2%}".format(RC)," | Fscore:{:.2%}".format(FS))
    print("False Positve Rate:{:.2%}".format(FPR), " | False Negative Rate:{:.2%}".format(FNR))
    print("Average Precision:{:.2%}".format(AP))
    print('RMSE : {:.4f}'.format(RMSE))

    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)

    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)

    #ROC
    plt.subplot(142)
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini), fontsize=10)
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
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP), fontsize=10)
    
    return ACC,PC,RC,FS,AP,roc_auc,gini,RMSE,FPR,FNR

def create_gainlift_table(data,var_to_rank,var_to_count_nonzero,n_qcut=10):
    tmpname = var_to_rank+'_tmp'
    data[tmpname] = pd.qcut(data[var_to_rank].rank(method='first'), n_qcut, labels=False)
    
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
    
    table_sorted['nonresponse'] = table_sorted[var_to_rank] - table_sorted[var_to_count_nonzero]
    table_sorted['cumulative_nonresponse'] = table_sorted['nonresponse'].cumsum()
    
    total_nonresponse = np.sum(table_sorted.loc[:,'nonresponse'])
    table_sorted['percent_of_nonevents'] = (table_sorted['nonresponse']/total_nonresponse)
    table_sorted['cumulative_percent_of_nonevents'] = table_sorted['percent_of_nonevents'].cumsum()
    
    total_response = np.sum(table_sorted.loc[:,var_to_count_nonzero])
    table_sorted['percent_of_events'] = (table_sorted[var_to_count_nonzero]/total_response)
    table_sorted['cumulative_percent_of_events'] = table_sorted['percent_of_events'].cumsum()
    
    table_sorted['cumulative_gain'] = (table_sorted['cumulative_response']/total_response)*100
    decile = np.linspace(1,n_qcut,n_qcut)
    table_sorted['decile'] = decile*(100/n_qcut)
    table_sorted['cumulative_lift'] = table_sorted['cumulative_gain']/(table_sorted['decile'])
    table_sorted.rename(columns={var_to_rank: 'counts'},inplace=True)
    table_sorted['ks'] = np.abs(table_sorted['cumulative_percent_of_events'] - table_sorted['cumulative_percent_of_nonevents'])
    table_sorted.index.names = ['rank']
    table_sorted['base_lift']=1
    table_sorted['base_gain']=[x for x in np.linspace(10,100,n_qcut)]
    
    data.drop(columns=[tmpname],inplace=True)
    
    return table_sorted

def print_gainlift_charts(table_sorted):
    """
        Program: print_gainlift_charts
        Author: Siraprapa W.
        
        Purpose:print gain lift charts   
    """
    fig = plt.figure(figsize=(20,4))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
   
    fig, axes = plt.subplots(nrows=1, ncols=2)
    table_sorted.plot(kind='line',x='decile',y=['cumulative_lift','base_lift'],style='o-',
                      ax=axes[0],figsize=(10,3),title='Cumulative Lift Curve')
    table_sorted.plot(kind='line',x='decile',y=['cumulative_gain','base_gain'],style='o-',
                      ax=axes[1],figsize=(10,3),title='Cumulative Gain Curve')
    
    # Adding data points over markers for Lift Curve
    for i, point in table_sorted.iterrows():
        axes[0].text(point['decile'], point['cumulative_lift'], "{:.1f}".format(point['cumulative_lift'], fontsize=8))
        #axes[0].text(point['decile'], point['base_lift'], "{:.1f}".format(point['base_lift']))

    # Adding data points over markers for Gain Curve
    for i, point in table_sorted.iterrows():
        axes[1].text(point['decile'], point['cumulative_gain'], "{:.1f}".format(point['cumulative_gain'], fontsize=8))
        #axes[1].text(point['decile'], point['base_gain'], "{:.1f}".format(point['base_gain']))

    #print(table_sorted.set_index('decile').loc[:,[var_to_count_nonzero,'cumulative_gain','cumulative_lift','ks']])

    
#

def display_gainlift_table(table_sorted,target='target'):
    display(table_sorted[['decile',target,'cumulative_gain','cumulative_lift','ks']].style.format({'ks':"{:.2%}",
                    'cumulative_lift':"{:.2f}",
                    'cumulative_gain':"{:.2f}",
                     'target':"{:.0f}",
                     'decile':"{:.0f}",
                      'index':False,
                    }))
    

def print_gainlift_table_and_charts(table_sorted,target='target'):
    # Set seaborn style
    sns.set()

    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # One row and three columns


    # Displaying table
    ax0 = plt.subplot(gs[0, 0])
    ax0.axis('tight')
    ax0.axis('off')
    table_to_display = table_sorted.loc[:, ['decile',target,'cumulative_gain','cumulative_lift', 'ks']]
    cell_text = []
    for row in range(len(table_to_display.index)):
        cell_text.append(table_to_display.iloc[row])

    # Formatting the data in table
    cell_text_format = [[
        "{:.0f}".format(x[0]),
        "{:.0f}".format(x[1]),
        "{:.2f}".format(x[2]),
        "{:.2f}".format(x[3]),
        "{:.2%}".format(x[4])
    ] for x in cell_text]

    # Creating the table and adding style
    table = ax0.table(cellText=cell_text_format, 
                      colLabels=table_to_display.columns, 
                      loc='center', 
                      cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 2)
    table.auto_set_column_width([0,1,2,3])

    for key, cell in table.get_celld().items():
        cell.set_edgecolor('black')
        if key[0] == 0:  # if this is a header cell
            cell.set_facecolor('#FF6347')  # set the header cell color to Tomato
            cell.get_text().set_color('white')  # set the header font color to white

    # Cumulative Lift Curve
    ax1 = plt.subplot(gs[0, 1])
    table_sorted.plot(kind='line', x='decile', y='cumulative_lift', style='o-', ax=ax1, title='Cumulative Lift Curve')
    ax1.axhline(1, color='red', linestyle='--',label='baseline')  # Flat line at lift=1
    for i, point in table_sorted.iterrows():
        ax1.text(point['decile'], point['cumulative_lift'], "{:.1f}".format(point['cumulative_lift']), fontsize=10)
    ax1.legend()
        
    # Cumulative Gain Curve
    # Append a row at the start for gain chart
    new_row = pd.DataFrame({"decile": 0, "cumulative_gain": 0}, index =[0])
    table_sorted = pd.concat([new_row, table_sorted]).reset_index(drop = True)
    
    ax2 = plt.subplot(gs[0, 2])
    table_sorted.plot(kind='line', x='decile', y='cumulative_gain', style='o-', ax=ax2, title='Cumulative Gain Curve')
    ax2.plot([0, 100], [0, 100], 'r--',label='baseline')  # Diagonal line from left to right
    for i, point in table_sorted.iterrows():
        ax2.text(point['decile'], point['cumulative_gain'], "{:.1f}".format(point['cumulative_gain']), fontsize=10)
    ax2.legend()
    
    plt.tight_layout(pad=1.0)
    plt.show()

    
# Generate missing report - that that we utilise applymap(set_nan) to the entire dataframe-series
def generate_miss_report(data):
    
    # Define missing/invalid patterns 
    ch_miss = ['n/a','na','na.','n.a.','n.a','*','-','unknown','email@domain.com',
               'testuser','u','99999999','null','none','c9999','z_error','z_missing','',' ','unspecified','nan']
    nm_miss = [99999999.0,-999.0]

    # define func for DataFrame.applymap
    def tuplizer(x):
        return tuple(x) if isinstance(x, (np.ndarray, list)) else x
    
    def set_nan(x):
        if type(x) is str :
            if x.lower() in ch_miss :
                return np.nan
            else:
                return x
        elif type(x) is int or type(x) is float:
            if x in nm_miss:
                return np.nan
            else:
                return x
        else:
             return x

    data_dub = data.applymap(set_nan)
    n_miss  = data_dub.isnull().sum() 
    p_miss  = round(data_dub.isnull().sum() / len(data_dub.index),2)
    n_unique = data_dub.nunique()
    #n_unique = data_dub.apply(tuplizer).nunique()
    #n_unique = data_dub.apply(lambda x: str(x)).nunique()
    p_unique = round(n_unique / len(data_dub.index),2)
    miss_report = pd.merge(n_miss.rename('n_miss'),p_miss.rename('%miss'),left_index=True,right_index=True)
    miss_report = miss_report.merge(n_unique.rename('n_unique'),left_index=True,right_index=True)
    miss_report = miss_report.merge(p_unique.rename('%unique'),left_index=True,right_index=True)
    miss_report['n_populated'] = len(data_dub.index) - miss_report['n_miss']
    miss_report['%populated'] = round(1.00 - miss_report['%miss'],2)
    miss_report.reset_index(inplace=True)
    miss_report.rename(columns={'index':'features'},inplace=True)
    return miss_report

class MetricContainer:
    def __init__(self):
        self.i=0
        self.model=[]
        self.desc=[]
        self.acc=[]
        self.pc=[]
        self.rc=[]
        self.fs=[]
        self.ap=[]
        self.rmse=[]
        self.auc=[]
        self.gini=[]
        self.fpr=[]
        self.fnr=[]
        self.clift10=[]
        self.clift20=[]
        self.clift30=[]
        
    def add(self,desc,acc,pc,rc,fs,ap,rmse,auc,gini,fpr,fnr,clift10,clift20,clift30):
        self.i+=1
        self.model.append("Model {}".format(self.i))
        self.desc.append(desc)
        self.acc.append(acc)
        self.pc.append(pc)
        self.rc.append(rc)
        self.fs.append(fs)
        self.ap.append(ap)
        self.rmse.append(rmse)
        self.auc.append(auc)
        self.gini.append(gini)
        self.fpr.append(fpr)
        self.fnr.append(fnr)
        self.clift10.append(clift10)
        self.clift20.append(clift20)
        self.clift30.append(clift30)
    
    def get(self):
        metric = pd.DataFrame([self.model,
                               self.desc,
                               self.acc,
                               self.pc,
                               self.rc,
                               self.fs,
                               self.ap,
                               self.rmse,
                               self.auc,
                               self.gini,
                               self.fpr,
                               self.fnr,
                               self.clift10,
                               self.clift20,
                               self.clift30,]).transpose()
        metric.columns=['model','desc','acc','pc','rc','fs','ap','rmse','auc','gini','fpr','fnr'
                        ,'clift10','clift20','clift30']
        metric.set_index('model')
        
        return metric

def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color