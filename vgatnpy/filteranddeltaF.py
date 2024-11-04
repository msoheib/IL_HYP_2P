#####################################################
#FILTER RAW TRACES and CALCULATE %DELTA/F
#####################################################
import pandas as pd
import helper_functions
import pandas as pd
import helper_functions

##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

def filter_and_calculate_deltaF(pathtracesexp, freq=29.87, cutofflow=0.08, cutoffhigh=5, percentile=0.08):
    
    #open and read dataframe with tracesRaw
    tracesRaw= pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.',engine='python')
    tracesRaw.set_index(['time'], inplace=True) 

    #select columns
    column_names = helper_functions.generate_column_names(0, 92)

    calcium_columns = tracesRaw[column_names]
    
    #FILTER TRACES BANDPASS
    filtered=helper_functions.bandpassfiltnewColumns(calcium_columns,freq=freq, cutofflow=cutofflow, cutoffhigh=cutoffhigh, butterord=2)

    #Calculate %deltaF/F
    deltaF, x_seconds =helper_functions.DeltaF_percentile_columns(filtered,freq,percentile)

    deltaF['pupil_size']=tracesRaw.pupil_size
    deltaF['speed']=tracesRaw.speed
    deltaF['degrees_180']=tracesRaw.degrees_180
    deltaF['degrees_225']=tracesRaw.degrees_225
    deltaF['degrees_135']=tracesRaw.degrees_135
    deltaF['degrees_0']=tracesRaw.degrees_0
    deltaF['degrees_315']=tracesRaw.degrees_315
    deltaF['degrees_90']=tracesRaw.degrees_90
    deltaF['degrees_270']=tracesRaw.degrees_270
    deltaF['degrees_45']=tracesRaw.degrees_45
    deltaF['direction']=tracesRaw.direction
    
    return deltaF

    #deltaF.to_csv('deltaF.csv', sep=',', index=True, header=True)
    
    #exit()
