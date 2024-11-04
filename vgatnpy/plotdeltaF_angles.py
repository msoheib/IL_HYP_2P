#####################################################
#plot deltaF and angles
#####################################################
import pandas as pd
import helper_functions
import pandas as pd
import helper_functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm

##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

#######################################
#PLOT
def plot_deltaF_angles(pathtracesexp, Select_columns=helper_functions.generate_column_names(0,10), timeWindowON=0, timeWindowOFF=400, condition_columns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315'], condition_colors=['black','blue','red','green','orange','magenta','brown','cyan'], scalebardeltaf=2, gaussian_sigma=30):
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    #open and read dataframe
    traces = pathtracesexp.copy()
    traces.set_index(['time'], inplace=True) 

    traceswindow = traces.loc[timeWindowON:timeWindowOFF, :]

    # Apply Gaussian smoothing with the specified sigma
    smoothed_traces = traceswindow[Select_columns].apply(lambda x: gaussian_filter1d(x, sigma=gaussian_sigma))

    figDelta = smoothed_traces.plot(subplots=True, sharey=True, legend=False, colormap=cm.winter)
    helper_functions.customize_pandas_plot(figDelta)

    [ax.vlines(x=timeWindowON-5,ymin=0,ymax=scalebardeltaf, label=str(scalebardeltaf)+'%', color="black", linewidth=2) for ax in figDelta.ravel()]
    [ax.axhline(y=0, color="black",alpha=0.5, linewidth=0.1) for ax in figDelta.ravel()]
    [ax.yaxis.set_visible(False)for ax in figDelta.ravel()]
    [ax.xaxis.set_visible(True)for ax in figDelta.ravel()]

    [ax.text(0, 0.1,  str(scalebardeltaf)+'%\n \u0394F/F\u2080', transform=ax.transAxes, ha='center', va='center', rotation='vertical', fontsize='x-small') for ax in figDelta.ravel()]

    n=0
    for ax in figDelta.ravel():
        ax.text(1,0.9 , Select_columns[n], ha='center', va='center', rotation='horizontal', fontsize='x-small', transform=ax.transAxes)
        n=n+1

    i=0
    xpos=0
    for cond in condition_columns:
        condition_index = traceswindow[condition_columns].loc[traceswindow[condition_columns][cond] == 1].index
        cond_max=condition_index.max()
        cond_min=condition_index.min()
        cond_x=traceswindow[cond].index.values
        cond_y=traceswindow[cond].values*smoothed_traces.max().max()
        #convert start and stop to zero for filling plot
        cond_y[0]=0
        cond_y[-1]=0

        [ax.fill(cond_x, cond_y, color=condition_colors[i], alpha=0.5, lw=0, label=cond)for ax in figDelta.ravel()]     
        plt.text(xpos, -1, cond, transform=ax.transAxes, ha='center', va='center', rotation='horizontal', fontsize='x-small',color=condition_colors[i], alpha=0.5)
        i=i+1
        xpos=xpos+0.15
    plt.show()