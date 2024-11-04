##############################################################################
#Open delta f data and calculate pearson correlation of rois with ori columns
#############################################################################
import pandas as pd
import helper_functions
import pandas as pd
import helper_functions
import matplotlib.pyplot as plt
##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

# openpathtracesexp= 'aligned_dff30.csv'
# corrthreshold=0.25
column_names = helper_functions.generate_column_names(0, 92)
##########################################################################
def calculate_correlation(df1, threshold=0.25, column_names = helper_functions.generate_column_names(0, 92)):
    
    df = df1.copy()
    df.set_index(['time'], inplace=True) 

    print(df)

    targetcorrelation=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315']

    correlationdata=pd.DataFrame()
    for c in targetcorrelation:
        correlation_degree =helper_functions.correlate_columns_with_target(df,c,column_names)
        correlationdata[c]=correlation_degree
        plt.plot(correlation_degree)
        plt.title(c)
        plt.show()

    correlationdata.to_csv('correlationbydegree.csv', sep=',', index = True, header=True)

    for c in targetcorrelation:
        correlatedorois=correlationdata[correlationdata[c]>threshold].index
        if  len(correlatedorois)>0:
            figcorrelated=df[correlatedorois].plot(subplots=True, sharey=False, legend=True, title=str(c))
            helper_functions.customize_pandas_plot(figcorrelated)
            plt.show()
        else:
            plt.text(0.2,0.5,"No correlation above threshold")
            plt.title(str(c))
            plt.show()
