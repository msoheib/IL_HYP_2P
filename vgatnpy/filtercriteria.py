#####################################################
# filtering by criteria, generates file with 'tracesStimuli.csv', 'pvalues_response.csv', and 'selected_rois.csv'
#####################################################
import pandas as pd
import helper_functions
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, zscore
import pandas as pd
import helper_functions
import numpy as np
from scipy.stats import ttest_1samp
import seaborn as sns
import plotly.graph_objects as go
##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

# pathtracesexp= 'aligned_dff30.csv'
# Select_columns=helper_functions.generate_column_names(0,92) #can replace here with an array with the roi names
condition_columns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315']
condition_colors=['black','blue','red','green','orange','magenta','brown','cyan']
# scalebardeltaf=1
# scalebarseconds=2
# secondsbefore=2
# secondsafter=6
# maxdeltaFrequired=0.3 #max of the mean threshold, equal or higher needed
def filter_criteria(pathtracesexp='', Select_columns=helper_functions.generate_column_names(0,92), condition_columns=condition_columns, condition_colors=condition_colors, scalebardeltaf=1, scalebarseconds=2, secondsbefore=2, secondsafter=6, maxdeltaFrequired=0.3):


    #Select_columns = [col for col in df.columns if col.isdigit()]
    #open and read dataframe
    traces1= pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.',engine='python')
    traces = traces1.copy()

    traces.set_index(['time'], inplace=True) 
    Select_columns = [col for col in traces.columns if col.isdigit()]
    #traces.drop(columns=['Unnamed: 0'], inplace=True)

    orieventsidx=helper_functions.getonsets(traces,condition_columns)

    timeonVal=[]
    for cond in condition_columns:
        timeon=orieventsidx[orieventsidx[cond]==1].index.tolist()
        timeonVal.append(timeon)

    onData=pd.DataFrame(np.transpose(timeonVal), columns=condition_columns)

    for o in onData.columns:
        for p in onData[o]:
           responsesatori=traces.loc[p-secondsbefore:p+secondsafter,:]


    tracesStimuli = pd.DataFrame()

    for o in onData.columns:
        rep=1
        for p in (onData[o].values.tolist()):
            segment = traces.loc[p-secondsbefore:p+secondsafter, :]

            # Transform the index to 5 decimals
            segment.index = np.round(segment.index - p, 5)
            segmentData=pd.DataFrame(np.transpose(segment))

            # Add new levels of index
            segmentData['degrees'] = o
            segmentData['rep'] = rep
            segmentData =  segmentData.set_index(['degrees', 'rep'], append=True)

            tracesStimuli=pd.concat([tracesStimuli,segmentData],sort=False)

            rep=rep+1

    tracesStimuli.to_csv('tracesStimuli.csv', sep=',', index = True, header=True)


    ############################################################################################
    #filter by criteria
    '''
        Apply criteria, p value <0.05, "paired ttest":
        One sample ttest to test whether the mean of the differences between the response after onset and before onset
        is greater than 0

        AND max deltaF threshold of the mean also required

        returns: pvalues_response.csv containing the p values for each roi and selected_rois.csv with the rois p <0.05at each orientation
    '''
    idx = pd.IndexSlice

    pvalues= []

    for i, cond in enumerate(condition_columns):
        for roi in Select_columns:
            justbeforezero=tracesStimuli.columns[tracesStimuli.columns.get_loc(0) - 1]
            before_0_means=tracesStimuli.loc[idx[[roi], [cond]], :justbeforezero].T.mean().values
            after_0_means=tracesStimuli.loc[idx[[roi], [cond]], 0:].T.mean().values
            maxdeltaF=tracesStimuli.loc[idx[[roi], [cond]], 0:].mean().max()
            diff = after_0_means - before_0_means
            t_stat, p_val = ttest_1samp(diff, popmean=0, alternative='greater')
            ttest=(cond,roi,p_val, maxdeltaF)
            pvalues.append(ttest)

    pvalues_response = pd.DataFrame(pvalues, columns=['ori','roi', 'p_val', 'maxdeltaF'])
    pvalues_response.to_csv('pvalues_response.csv', sep=',', index = True, header=True)   

    #filter by p value and deltaF max
    significantpval = pvalues_response[(pvalues_response['p_val'] < 0.05) & (pvalues_response['maxdeltaF'] >= maxdeltaFrequired)]


    ori_rois= []
    for cond in condition_columns:
        sigroi=(cond, significantpval[significantpval['ori']==cond]['roi'].values)
        ori_rois.append(sigroi)
    sigroi=("atleast_oneori", significantpval['roi'].unique())
    ori_rois.append(sigroi)

    selected_rois = pd.DataFrame(ori_rois, columns=['ori','rois'])
    print(selected_rois)
    selected_rois.to_csv('selected_rois.csv', sep=',', index = True, header=True)


    ####################################################################################################################
    #PLOT

    tracesStimuli.columns= pd.to_numeric(tracesStimuli.columns, errors='coerce')
    atleastoneori=selected_rois[selected_rois['ori']=='atleast_oneori']['rois'].values[0]


    for roi in atleastoneori:
        fig, axsfig = plt.subplots(1, len(condition_columns),figsize=(8, 2), sharey=True, sharex=True)
        for i, cond in enumerate(condition_columns):

            tracesStimuli.loc[idx[[roi], [cond]], :].T.plot(ax=axsfig[i], color='black', alpha=0.2, lw=1,legend=False,)
            tracesStimuli.loc[idx[[roi], [cond]], :].mean().T.plot(ax=axsfig[i], title=roi+'\n'+cond, color=condition_colors[i],lw=2,legend=False)
            if roi in selected_rois[selected_rois['ori'] == cond]['rois'].values[0]:
                axsfig[i].text(0.9, 0.9, '*', transform=axsfig[i].transAxes)

        [ax.title.set_size('x-small')for ax in axsfig]
        [ax.vlines(x=-secondsbefore-0.5, ymin=0, ymax=scalebardeltaf,label=str(scalebardeltaf), color="black", linewidth=2) for ax in axsfig]
        [ax.axvline(x=0,linewidth=0.5, color='black', alpha=0.5) for ax in axsfig]
        [ax.axhline(y=0, color="black",alpha=0.5, linewidth=0.2) for ax in axsfig]
        [ax.yaxis.set_visible(False)for ax in axsfig]
        [ax.xaxis.set_visible(True)for ax in axsfig]
        [ax.set_frame_on(False)for ax in axsfig]
        [ax.text(-secondsbefore-0.8, 0.1,  str(scalebardeltaf)+'% \u0394F/F%\u2080',  ha='center', va='center', rotation='vertical', fontsize='x-small') for ax in axsfig]


        plt.tight_layout()
        plt.show()
##########################################33


# def filter_criteria_heatmap(pathtracesexp='aligned_dff30.csv', Select_columns=helper_functions.generate_column_names(0,92), condition_columns=condition_columns, condition_colors=condition_colors, scalebardeltaf=1, scalebarseconds=2, secondsbefore=2, secondsafter=6, maxdeltaFrequired=0.3):
     
#     #open and read dataframe
#     traces1= pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.',engine='python')
#     traces = traces1.copy()

#     traces.set_index(['time'], inplace=True) 
#     #traces.drop(columns=['Unnamed: 0'], inplace=True)

#     orieventsidx=helper_functions.getonsets(traces,condition_columns)

#     timeonVal=[]
#     for cond in condition_columns:
#         timeon=orieventsidx[orieventsidx[cond]==1].index.tolist()
#         timeonVal.append(timeon)

#     onData=pd.DataFrame(np.transpose(timeonVal), columns=condition_columns)

#     for o in onData.columns:
#         for p in onData[o]:
#            responsesatori=traces.loc[p-secondsbefore:p+secondsafter,:]


#     tracesStimuli = pd.DataFrame()

#     for o in onData.columns:
#         rep=1
#         for p in (onData[o].values.tolist()):
#             segment = traces.loc[p-secondsbefore:p+secondsafter, :]

#             # Transform the index to 5 decimals
#             segment.index = np.round(segment.index - p, 5)
#             segmentData=pd.DataFrame(np.transpose(segment))

#             # Add new levels of index
#             segmentData['degrees'] = o
#             segmentData['rep'] = rep
#             segmentData =  segmentData.set_index(['degrees', 'rep'], append=True)

#             tracesStimuli=pd.concat([tracesStimuli,segmentData],sort=False)

#             rep=rep+1

#     tracesStimuli.to_csv('tracesStimuli.csv', sep=',', index = True, header=True)


#     ############################################################################################
#     #filter by criteria
#     '''
#         Apply criteria, p value <0.05, "paired ttest":
#         One sample ttest to test whether the mean of the differences between the response after onset and before onset
#         is greater than 0

#         AND max deltaF threshold of the mean also required

#         returns: pvalues_response.csv containing the p values for each roi and selected_rois.csv with the rois p <0.05at each orientation
#     '''
#     idx = pd.IndexSlice

#     pvalues= []

#     for i, cond in enumerate(condition_columns):
#         for roi in Select_columns:
#             justbeforezero=tracesStimuli.columns[tracesStimuli.columns.get_loc(0) - 1]
#             before_0_means=tracesStimuli.loc[idx[[roi], [cond]], :justbeforezero].T.mean().values
#             after_0_means=tracesStimuli.loc[idx[[roi], [cond]], 0:].T.mean().values
#             maxdeltaF=tracesStimuli.loc[idx[[roi], [cond]], 0:].mean().max()
#             diff = after_0_means - before_0_means
#             t_stat, p_val = ttest_1samp(diff, popmean=0, alternative='greater')
#             ttest=(cond,roi,p_val, maxdeltaF)
#             pvalues.append(ttest)

#     pvalues_response = pd.DataFrame(pvalues, columns=['ori','roi', 'p_val', 'maxdeltaF'])
#     pvalues_response.to_csv('pvalues_response.csv', sep=',', index = True, header=True)   

#     #filter by p value and deltaF max
#     significantpval = pvalues_response[(pvalues_response['p_val'] < 0.05) & (pvalues_response['maxdeltaF'] >= maxdeltaFrequired)]


#     ori_rois= []
#     for cond in condition_columns:
#         sigroi=(cond, significantpval[significantpval['ori']==cond]['roi'].values)
#         ori_rois.append(sigroi)
#     sigroi=("atleast_oneori", significantpval['roi'].unique())
#     ori_rois.append(sigroi)

#     selected_rois = pd.DataFrame(ori_rois, columns=['ori','rois'])
#     print(selected_rois)
#     selected_rois.to_csv('selected_rois.csv', sep=',', index = True, header=True)


#     ####################################################################################################################
#     #PLOT

#     tracesStimuli.columns= pd.to_numeric(tracesStimuli.columns, errors='coerce')
#     atleastoneori=selected_rois[selected_rois['ori']=='atleast_oneori']['rois'].values[0]

#     # Create transposed heatmaps (switch x and y axis) with adjusted subplot size
#     for roi in atleastoneori:
#         # Adjust the figsize to make subplots bigger
#         fig, axsfig = plt.subplots(1, len(condition_columns), figsize=(16, 4), sharey=True, sharex=True)  # Increased width from 8 to 16
#         for i, cond in enumerate(condition_columns):
#             data = tracesStimuli.loc[idx[[roi], [cond]], :].T
#             # Transpose the data for plotting to switch axes
#             sns.heatmap(data.transpose(), ax=axsfig[i], cmap='viridis', cbar=i == 0, cbar_kws={'label': '% \u0394F/F%\u2080'})
#             axsfig[i].set_title(roi + '\n' + cond, fontsize='x-small')
#             if roi in selected_rois[selected_rois['ori'] == cond]['rois'].values[0]:
#                 axsfig[i].text(0.9, 0.9, '*', transform=axsfig[i].transAxes, fontsize='medium', color='red')

#         # Adjust layout and plot settings
#         for ax in axsfig:
#             ax.set_ylabel('Time (s)')
#             ax.axhline(y=secondsbefore + 0.5, linewidth=0.5, color='black', alpha=0.5)  # Time of event
#             if ax is axsfig[0]:
#                 ax.set_xlabel('Signal Intensity')
#             ax.yaxis.set_visible(True)
#             ax.set_frame_on(True)

#         plt.subplots_adjust(wspace=0.3)  # Adjust the spacing between subplots
#         plt.tight_layout()
#         plt.show()

#     # # Create one heatmap per condition with means across trials for all ROIs
#     # for cond in condition_columns:
#     #     # Aggregate data across trials to get the mean for each ROI at each time point
#     #     aggregated_data = tracesStimuli.xs(cond, level='degrees').groupby(level='roi').mean().T

#     #     # Plotting the heatmap for the condition
#     #     fig, ax = plt.subplots(figsize=(16, 4))  # Set the size of the figure
#     #     sns.heatmap(aggregated_data, ax=ax, cmap='viridis', cbar_kws={'label': '% \u0394F/F%\u2080'})
#     #     ax.set_title(f'Average Response for {cond}', fontsize='x-small')
#     #     ax.set_ylabel('Time (s)')
#     #     ax.set_xlabel('ROI')
        
#     #     # Draw a line to indicate the event time
#     #     event_time_index = secondsbefore + 0.5  # Adjust this if your time index is different
#     #     ax.axhline(y=event_time_index, color='black', linewidth=0.5, alpha=0.5)  # Time of event

#     #     plt.tight_layout()
#     #     plt.show()



def criteria_plot_population_response_old(pathtracesexp='aligned_dff30.csv', 
                             condition_columns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315'],
                             condition_colors=['black','blue','red','green','orange','magenta','brown','cyan'],
                             secondsbefore=2, secondsafter=6, inclusion_criterion='IC1', interp_hz=10, p_threshold=0.05, maxdeltaFrequired=0.3, z_threshold=3):
    
    # Load data
    if isinstance(pathtracesexp, pd.DataFrame):
        traces_codex = pathtracesexp
    else:
        traces_codex = pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.', engine='python')
    traces = traces_codex.copy()
    traces_orig = traces.copy()
    
    # Identify ROI columns and non-ROI columns
    roi_columns = [col for col in traces.columns if col.isdigit()]
    non_roi_columns = [col for col in traces.columns if not col.isdigit()]
    
    traces.set_index(['time'], inplace=True)
    
    # Get stimulus onset times
    orieventsidx = helper_functions.getonsets(traces, condition_columns)
    
    # Prepare data structure for storing responses
    all_responses = {cond: [] for cond in condition_columns}
    
    # Process each ROI
    windows = []
    for roi in roi_columns:
        roi_data = traces[roi]
        roi_responses = []
        for cond in condition_columns:
            onset_times = orieventsidx[orieventsidx[cond]==1].index
            
            # Extract response windows
            responses = np.array([roi_data.loc[t-secondsbefore:t+secondsafter].values for t in onset_times])
            roi_responses.append(responses)
            
            # Store responses
            all_responses[cond].append(responses)
        
        windows.append(np.concatenate(roi_responses))
    
    windows = np.array(windows)
    windows = np.transpose(windows, (1, 0, 2))  # Reshape to (trials, ROIs, time)
    
    # Compute baseline and evoked responses
    stimulus_onset_idx = int(secondsbefore * interp_hz)
    baseline = windows[:,:,:stimulus_onset_idx]
    evoked_responses = windows[:,:,stimulus_onset_idx:]
    
    # Apply selected inclusion criterion
    if inclusion_criterion is None:
        selected_rois = np.arange(len(roi_columns))
    elif inclusion_criterion == 'IC1':
        mean_evoked_responses = np.mean(evoked_responses, axis=0)
        max_mean_evoked_responses = np.max(mean_evoked_responses, axis=1)
        selected_rois = np.where(max_mean_evoked_responses > 10)[0]
    elif inclusion_criterion == 'IC2':
        all_baseline_sds = np.std(baseline, axis=2)
        max_responses = np.max(evoked_responses, axis=2)
        deviant_responses = max_responses > 3*all_baseline_sds
        large_responses = max_responses > 5
        sig_responses = deviant_responses & large_responses
        half_trials = sig_responses.shape[0] / 2
        selected_rois = np.where(np.sum(sig_responses, axis=0) > half_trials)[0]
    elif inclusion_criterion == 'IC3':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        n = mean_trial_responses.shape[0]
        t, p = ttest_rel(mean_trial_responses, mean_trial_baselines)
        selected_rois = np.where(p < 0.05 / n)[0]
    elif inclusion_criterion == 'IC4':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        response_means = np.mean(mean_trial_responses, axis=0)
        response_sds = np.std(mean_trial_responses, axis=0)
        mean_trial_baselines = np.mean(baseline, axis=2)
        baseline_means = np.mean(mean_trial_baselines, axis=0)
        baseline_sds = np.std(mean_trial_baselines, axis=0)
        reliabilities = (response_means - baseline_means) / (response_sds + baseline_sds)
        selected_rois = np.where((reliabilities > 1) & (response_means > 6))[0]
    elif inclusion_criterion == 'IC5':
        max_responses = np.max(evoked_responses, axis=(0,2))
        selected_rois = np.where(max_responses > 4)[0]
    elif inclusion_criterion == 'IC6':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        t, p = ttest_rel(mean_trial_baselines, mean_trial_responses)
        max_responses = np.max(np.mean(evoked_responses, axis=0), axis=1)
        selected_rois = np.where((p < p_threshold) & (max_responses >= maxdeltaFrequired))[0]
    elif inclusion_criterion == 'IC7':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        z_scores = np.mean((mean_trial_responses - np.mean(mean_trial_baselines, axis=0)) / np.std(mean_trial_baselines, axis=0), axis=0)
        selected_rois = np.where(z_scores > z_threshold)[0]
    elif inclusion_criterion == 'IC8':
        # Original criteria
        pvalues = []
        for i, cond in enumerate(condition_columns):
            for roi in Select_columns:
                justbeforezero = tracesStimuli.columns[tracesStimuli.columns.get_loc(0) - 1]
                before_0_means = tracesStimuli.loc[idx[roi, cond], :justbeforezero].T.mean().values
                after_0_means = tracesStimuli.loc[idx[roi, cond], 0:].T.mean().values
                maxdeltaF = tracesStimuli.loc[idx[roi, cond], 0:].mean().max()
                diff = after_0_means - before_0_means
                t_stat, p_val = ttest_1samp(diff, popmean=0, alternative='greater')
                ttest = (cond, roi, p_val, maxdeltaF)
                pvalues.append(ttest)

        pvalues_response = pd.DataFrame(pvalues, columns=['ori', 'roi', 'p_val', 'maxdeltaF'])
        pvalues_response.to_csv('pvalues_response.csv', sep=',', index=True, header=True)   

        significantpval = pvalues_response[(pvalues_response['p_val'] < p_threshold) & (pvalues_response['maxdeltaF'] >= maxdeltaFrequired)]
        selected_rois = significantpval['roi'].unique()
    else:
        raise ValueError("Invalid inclusion criterion. Choose from 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', or 'IC7'.")
    
    # Prepare time axis
    time = np.linspace(-secondsbefore, secondsafter, windows.shape[2])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot population response lines
    for cond, color in zip(condition_columns, condition_colors):
        responsive_data = [resp for roi, resp in zip(range(len(roi_columns)), all_responses[cond]) if roi in selected_rois]
        if responsive_data:
            mean_response = np.mean(np.mean(responsive_data, axis=0), axis=0)
            sem_response = np.std(np.mean(responsive_data, axis=0), axis=0) / np.sqrt(len(responsive_data))
            
            ax1.plot(time, mean_response, color=color, label=cond)
            ax1.fill_between(time, mean_response-sem_response, mean_response+sem_response, color=color, alpha=0.2)

    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mean ΔF/F%')
    ax1.set_title(f'Population Response of Stimuli-Responsive ROIs\n{inclusion_criterion} (n={len(selected_rois)})')
    ax1.legend()


    #####################
    
    # Prepare data for condition heatmap
    heatmap_data = []
    for cond in condition_columns:
        responsive_data = [resp for roi, resp in zip(range(len(roi_columns)), all_responses[cond]) if roi in selected_rois]
        if responsive_data:
            mean_response = np.mean(np.mean(responsive_data, axis=0), axis=0)
            heatmap_data.append(mean_response)
    
    heatmap_data = np.array(heatmap_data)
    
    # Plot condition heatmap
    sns.heatmap(heatmap_data, ax=ax2, cmap='viridis', cbar_kws={'label': 'Mean ΔF/F%'})
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Condition')
    ax2.set_title('Mean Population Response Heatmap')
    ax2.set_xticks(np.linspace(0, len(time)-1, 5))
    ax2.set_xticklabels([f'{t:.1f}' for t in np.linspace(time[0], time[-1], 5)])
    ax2.set_yticks(np.arange(len(condition_columns)) + 0.5)
    ax2.set_yticklabels(condition_columns)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of responsive ROIs: {len(selected_rois)}")
    print(f"Responsive ROIs: {sorted(selected_rois)}")
    
   # Prepare data for averaged ROI heatmap
    roi_heatmap_data = []
    for roi in selected_rois:
        roi_responses = []
        for cond in condition_columns:
            roi_responses.extend(all_responses[cond][roi])
        mean_response = np.mean(roi_responses, axis=0)
        roi_heatmap_data.append(mean_response)
    
    roi_heatmap_data = np.array(roi_heatmap_data)
    
    # Scale the data to 0-100 range
    min_val = np.min(roi_heatmap_data)
    max_val = np.max(roi_heatmap_data)
    roi_heatmap_data_scaled = (roi_heatmap_data - min_val) / (max_val - min_val) * 100
    
    # Find the time of peak response after stimulus onset for each ROI
    stimulus_onset_idx = np.argmin(np.abs(time))  # Find the index closest to time 0
    peak_times = np.argmax(roi_heatmap_data_scaled[:, stimulus_onset_idx:], axis=1) + stimulus_onset_idx
    
    # Sort ROIs based on their peak response time (in reverse order)
    sorted_indices = np.argsort(peak_times)[::-1]  # Reverse the order
    roi_heatmap_data_sorted = roi_heatmap_data_scaled[sorted_indices]
    sorted_roi_labels = [f'ROI {selected_rois[i]}' for i in sorted_indices]
    
    # Create compact Plotly heatmap with reverse-sorted ROIs and fixed color scale
    plotly_fig = go.Figure(data=go.Heatmap(
        z=roi_heatmap_data_sorted,
        x=time,
        y=sorted_roi_labels,
        colorscale='Viridis',
        zmin=0,
        zmax=100,
        colorbar=dict(title='%ΔF/F%', titleside='right', thickness=15, len=0.9)
    ))
    
    plotly_fig.update_layout(
        title=dict(
            text=f'Sorted ROI Responses ({inclusion_criterion}, n={len(selected_rois)})',
            font=dict(size=14),
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Time (s)',
        yaxis_title='ROI (sorted by peak time, latest to earliest)',
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis=dict(tickmode='array', tickvals=np.arange(-secondsbefore, secondsafter+1)),
        yaxis=dict(tickmode='array', tickvals=np.arange(0, len(selected_rois), max(1, len(selected_rois)//10))),
    )
    
    # Add vertical line at stimulus onset
    plotly_fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=1,
        yref="paper",
        line=dict(color="white", width=2)
    )
    
    plotly_fig.show()
    
    print(f"Number of responsive ROIs: {len(selected_rois)}")
    print(f"Responsive ROIs (sorted by peak time, latest to earliest): {[selected_rois[i] for i in sorted_indices]}")
    print(f"Original ΔF/F% range: {min_val:.4f} to {max_val:.4f}")

    # Create DataFrame with responsive neurons and non-ROI columns
    selected_roi_columns = [roi_columns[i] for i in selected_rois]
    responsive_df = traces_orig[selected_roi_columns + non_roi_columns]
    return responsive_df, plotly_fig



##########
def filter_criteria(pathtracesexp='', Select_columns=None, condition_columns=None, condition_colors=None, scalebardeltaf=1, scalebarseconds=2, secondsbefore=2, secondsafter=6, maxdeltaFrequired=0.3, inclusion_criterion='IC8', p_threshold=0.05, z_threshold=3):

    # Open and read dataframe
    if isinstance(pathtracesexp, pd.DataFrame):
        traces = pathtracesexp.copy()
    else:
        traces = pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.', engine='python')

    # Check if 'time' column exists, if not, create one
    if 'time' not in traces.columns:
        traces['time'] = np.arange(len(traces)) / 10  # Assuming 10 Hz sampling rate

    traces.set_index(['time'], inplace=True)

    # If Select_columns is not provided, use all numeric columns
    if Select_columns is None:
        Select_columns = traces.select_dtypes(include=[np.number]).columns.tolist()

    # If condition_columns is not provided, try to infer from column names
    if condition_columns is None:
        condition_columns = [col for col in traces.columns if col.startswith('degrees_')]

    # If condition_colors is not provided, generate default colors
    if condition_colors is None:
        condition_colors = plt.cm.rainbow(np.linspace(0, 1, len(condition_columns)))

    # Check if all condition columns exist in the DataFrame
    missing_columns = [col for col in condition_columns if col not in traces.columns]
    if missing_columns:
        raise ValueError(f"The following columns are missing from the DataFrame: {missing_columns}")

    orieventsidx = helper_functions.getonsets(traces, condition_columns)

    timeonVal = []
    for cond in condition_columns:
        timeon = orieventsidx[orieventsidx[cond]==1].index.tolist()
        timeonVal.append(timeon)

    onData = pd.DataFrame(np.transpose(timeonVal), columns=condition_columns)

    tracesStimuli = pd.DataFrame()

    for o in onData.columns:
        rep = 1
        for p in (onData[o].values.tolist()):
            segment = traces.loc[p-secondsbefore:p+secondsafter, Select_columns]
            segment.index = np.round(segment.index - p, 5)
            segmentData = pd.DataFrame(np.transpose(segment))
            segmentData['degrees'] = o
            segmentData['rep'] = rep
            segmentData = segmentData.set_index(['degrees', 'rep'], append=True)
            tracesStimuli = pd.concat([tracesStimuli, segmentData], sort=False)
            rep = rep + 1

    tracesStimuli.to_csv('tracesStimuli.csv', sep=',', index=True, header=True)

    # Define idx for multi-index selection
    idx = pd.IndexSlice

    # Prepare data for inclusion criteria
    windows = []
    for roi in Select_columns:
        roi_data = tracesStimuli.loc[idx[roi, :, :], :]
        roi_responses = []
        for cond in condition_columns:
            responses = roi_data.loc[idx[:, cond, :], :].values
            roi_responses.append(responses)
        windows.append(np.concatenate(roi_responses))
    
    windows = np.array(windows)
    windows = np.transpose(windows, (1, 0, 2))  # Reshape to (trials, ROIs, time)
    
    # Compute baseline and evoked responses
    stimulus_onset_idx = int(secondsbefore * 10)  # Assuming 10 Hz sampling rate
    baseline = windows[:,:,:stimulus_onset_idx]
    evoked_responses = windows[:,:,stimulus_onset_idx:]

    # Apply selected inclusion criterion
    if inclusion_criterion == 'IC1':
        mean_evoked_responses = np.mean(evoked_responses, axis=0)
        max_mean_evoked_responses = np.max(mean_evoked_responses, axis=1)
        selected_rois = np.where(max_mean_evoked_responses > 10)[0]
    elif inclusion_criterion == 'IC2':
        all_baseline_sds = np.std(baseline, axis=2)
        max_responses = np.max(evoked_responses, axis=2)
        deviant_responses = max_responses > 3*all_baseline_sds
        large_responses = max_responses > 5
        sig_responses = deviant_responses & large_responses
        half_trials = sig_responses.shape[0] / 2
        selected_rois = np.where(np.sum(sig_responses, axis=0) > half_trials)[0]
    elif inclusion_criterion == 'IC3':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        n = mean_trial_responses.shape[0]
        t, p = ttest_rel(mean_trial_responses, mean_trial_baselines)
        selected_rois = np.where(p < 0.05 / n)[0]
    elif inclusion_criterion == 'IC4':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        response_means = np.mean(mean_trial_responses, axis=0)
        response_sds = np.std(mean_trial_responses, axis=0)
        mean_trial_baselines = np.mean(baseline, axis=2)
        baseline_means = np.mean(mean_trial_baselines, axis=0)
        baseline_sds = np.std(mean_trial_baselines, axis=0)
        reliabilities = (response_means - baseline_means) / (response_sds + baseline_sds)
        selected_rois = np.where((reliabilities > 1) & (response_means > 6))[0]
    elif inclusion_criterion == 'IC5':
        max_responses = np.max(evoked_responses, axis=(0,2))
        selected_rois = np.where(max_responses > 4)[0]
    elif inclusion_criterion == 'IC6':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        t, p = ttest_rel(mean_trial_baselines, mean_trial_responses)
        max_responses = np.max(np.mean(evoked_responses, axis=0), axis=1)
        selected_rois = np.where((p < p_threshold) & (max_responses >= maxdeltaFrequired))[0]
    elif inclusion_criterion == 'IC7':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        z_scores = np.mean((mean_trial_responses - np.mean(mean_trial_baselines, axis=0)) / np.std(mean_trial_baselines, axis=0), axis=0)
        selected_rois = np.where(z_scores > z_threshold)[0]
    elif inclusion_criterion == 'IC8':
        # Original criteria
        pvalues = []
        for i, cond in enumerate(condition_columns):
            for roi in Select_columns:
                justbeforezero = tracesStimuli.columns[tracesStimuli.columns.get_loc(0) - 1]
                before_0_means = tracesStimuli.loc[idx[roi, cond], :justbeforezero].T.mean().values
                after_0_means = tracesStimuli.loc[idx[roi, cond], 0:].T.mean().values
                maxdeltaF = tracesStimuli.loc[idx[roi, cond], 0:].mean().max()
                diff = after_0_means - before_0_means
                t_stat, p_val = ttest_1samp(diff, popmean=0, alternative='greater')
                ttest = (cond, roi, p_val, maxdeltaF)
                pvalues.append(ttest)

        pvalues_response = pd.DataFrame(pvalues, columns=['ori', 'roi', 'p_val', 'maxdeltaF'])
        pvalues_response.to_csv('pvalues_response.csv', sep=',', index=True, header=True)   

        significantpval = pvalues_response[(pvalues_response['p_val'] < p_threshold) & (pvalues_response['maxdeltaF'] >= maxdeltaFrequired)]
        selected_rois = significantpval['roi'].unique()
    elif inclusion_criterion == None:
        selected_rois = np.arange(len(Select_columns))
    else:
        raise ValueError("Invalid inclusion criterion. Choose from 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', 'IC7', or 'IC8'.")

    # Create selected_rois DataFrame
    ori_rois = []
    for cond in condition_columns:
        if inclusion_criterion == 'IC8':
            sigroi = (cond, [roi for roi in selected_rois if roi in significantpval[significantpval['ori']==cond]['roi'].values])
        else:
            sigroi = (cond, [Select_columns[i] for i in selected_rois])
        ori_rois.append(sigroi)
    sigroi = ("atleast_oneori", [Select_columns[i] for i in selected_rois] if inclusion_criterion != 'IC8' else selected_rois)
    ori_rois.append(sigroi)

    selected_rois_df = pd.DataFrame(ori_rois, columns=['ori', 'rois'])
    print(selected_rois_df)
    selected_rois_df.to_csv('selected_rois.csv', sep=',', index=True, header=True)

    # Plotting (unchanged)
    tracesStimuli.columns = pd.to_numeric(tracesStimuli.columns, errors='coerce')
    atleastoneori = selected_rois_df[selected_rois_df['ori']=='atleast_oneori']['rois'].values[0]

    for roi in atleastoneori:
        fig, axsfig = plt.subplots(1, len(condition_columns), figsize=(8, 2), sharey=True, sharex=True)
        for i, cond in enumerate(condition_columns):
            tracesStimuli.loc[idx[roi, cond], :].T.plot(ax=axsfig[i], color='black', alpha=0.2, lw=1, legend=False)
            tracesStimuli.loc[idx[roi, cond], :].mean().T.plot(ax=axsfig[i], title=str(roi)+'\n'+cond, color=condition_colors[i], lw=2, legend=False)
            if roi in selected_rois_df[selected_rois_df['ori'] == cond]['rois'].values[0]:
                axsfig[i].text(0.9, 0.9, '*', transform=axsfig[i].transAxes)

        [ax.title.set_size('x-small') for ax in axsfig]
        [ax.vlines(x=-secondsbefore-0.5, ymin=0, ymax=scalebardeltaf, label=str(scalebardeltaf), color="black", linewidth=2) for ax in axsfig]
        [ax.axvline(x=0, linewidth=0.5, color='black', alpha=0.5) for ax in axsfig]
        [ax.axhline(y=0, color="black", alpha=0.5, linewidth=0.2) for ax in axsfig]
        [ax.yaxis.set_visible(False) for ax in axsfig]
        [ax.xaxis.set_visible(True) for ax in axsfig]
        [ax.set_frame_on(False) for ax in axsfig]
        [ax.text(-secondsbefore-0.8, 0.1, str(scalebardeltaf)+'% \u0394F/F%\u2080', ha='center', va='center', rotation='vertical', fontsize='x-small') for ax in axsfig]

        plt.tight_layout()
        plt.show()

    return selected_rois_df, tracesStimuli







def criteria_plot_population_response(pathtracesexp='aligned_dff30.csv', 
                             condition_columns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315'],
                             condition_colors=['black','blue','red','green','orange','magenta','brown','cyan'],
                             secondsbefore=2, secondsafter=6, inclusion_criterion='IC1', interp_hz=10, p_threshold=0.05, maxdeltaFrequired=0.3, z_threshold=3, reorder_rois=True):
    
    # Load data
    if isinstance(pathtracesexp, pd.DataFrame):
        traces_codex = pathtracesexp
    else:
        traces_codex = pd.read_csv(pathtracesexp, delimiter=",", header=0, decimal='.', engine='python')
    traces = traces_codex.copy()
    traces_orig = traces.copy()
    
    # Identify ROI columns and non-ROI columns
    roi_columns = [col for col in traces.columns if col.isdigit()]
    non_roi_columns = [col for col in traces.columns if not col.isdigit()]
    
    traces.set_index(['time'], inplace=True)
    
    # Get stimulus onset times
    orieventsidx = helper_functions.getonsets(traces, condition_columns)
    
    # Prepare data structure for storing responses
    all_responses = {cond: [] for cond in condition_columns}
    
    # Process each ROI
    windows = []
    for roi in roi_columns:
        roi_data = traces[roi]
        roi_responses = []
        for cond in condition_columns:
            onset_times = orieventsidx[orieventsidx[cond]==1].index
            
            # Extract response windows
            responses = np.array([roi_data.loc[t-secondsbefore:t+secondsafter].values for t in onset_times])
            roi_responses.append(responses)
            
            # Store responses
            all_responses[cond].append(responses)
        
        windows.append(np.concatenate(roi_responses))
    
    windows = np.array(windows)
    windows = np.transpose(windows, (1, 0, 2))  # Reshape to (trials, ROIs, time)
    
    # Compute baseline and evoked responses
    stimulus_onset_idx = int(secondsbefore * interp_hz)
    baseline = windows[:,:,:stimulus_onset_idx]
    evoked_responses = windows[:,:,stimulus_onset_idx:]
    
    # Apply selected inclusion criterion
    if inclusion_criterion is None:
        selected_rois = np.arange(len(roi_columns))
    elif inclusion_criterion == 'IC1':
        mean_evoked_responses = np.mean(evoked_responses, axis=0)
        max_mean_evoked_responses = np.max(mean_evoked_responses, axis=1)
        selected_rois = np.where(max_mean_evoked_responses > 10)[0]
    elif inclusion_criterion == 'IC2':
        all_baseline_sds = np.std(baseline, axis=2)
        max_responses = np.max(evoked_responses, axis=2)
        deviant_responses = max_responses > 3*all_baseline_sds
        large_responses = max_responses > 5
        sig_responses = deviant_responses & large_responses
        half_trials = sig_responses.shape[0] / 2
        selected_rois = np.where(np.sum(sig_responses, axis=0) > half_trials)[0]
    elif inclusion_criterion == 'IC3':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        n = mean_trial_responses.shape[0]
        t, p = ttest_rel(mean_trial_responses, mean_trial_baselines)
        selected_rois = np.where(p < 0.05 / n)[0]
    elif inclusion_criterion == 'IC4':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        response_means = np.mean(mean_trial_responses, axis=0)
        response_sds = np.std(mean_trial_responses, axis=0)
        mean_trial_baselines = np.mean(baseline, axis=2)
        baseline_means = np.mean(mean_trial_baselines, axis=0)
        baseline_sds = np.std(mean_trial_baselines, axis=0)
        reliabilities = (response_means - baseline_means) / (response_sds + baseline_sds)
        selected_rois = np.where((reliabilities > 1) & (response_means > 6))[0]
    elif inclusion_criterion == 'IC5':
        max_responses = np.max(evoked_responses, axis=(0,2))
        selected_rois = np.where(max_responses > 4)[0]
    elif inclusion_criterion == 'IC6':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        t, p = ttest_rel(mean_trial_baselines, mean_trial_responses)
        max_responses = np.max(np.mean(evoked_responses, axis=0), axis=1)
        selected_rois = np.where((p < p_threshold) & (max_responses >= maxdeltaFrequired))[0]
    elif inclusion_criterion == 'IC7':
        mean_trial_responses = np.mean(evoked_responses, axis=2)
        mean_trial_baselines = np.mean(baseline, axis=2)
        z_scores = np.mean((mean_trial_responses - np.mean(mean_trial_baselines, axis=0)) / np.std(mean_trial_baselines, axis=0), axis=0)
        selected_rois = np.where(z_scores > z_threshold)[0]
    elif inclusion_criterion == 'IC8':
        # Original criteria
        pvalues = []
        for i, cond in enumerate(condition_columns):
            for roi in Select_columns:
                justbeforezero = tracesStimuli.columns[tracesStimuli.columns.get_loc(0) - 1]
                before_0_means = tracesStimuli.loc[idx[roi, cond], :justbeforezero].T.mean().values
                after_0_means = tracesStimuli.loc[idx[roi, cond], 0:].T.mean().values
                maxdeltaF = tracesStimuli.loc[idx[roi, cond], 0:].mean().max()
                diff = after_0_means - before_0_means
                t_stat, p_val = ttest_1samp(diff, popmean=0, alternative='greater')
                ttest = (cond, roi, p_val, maxdeltaF)
                pvalues.append(ttest)

        pvalues_response = pd.DataFrame(pvalues, columns=['ori', 'roi', 'p_val', 'maxdeltaF'])
        pvalues_response.to_csv('pvalues_response.csv', sep=',', index=True, header=True)   

        significantpval = pvalues_response[(pvalues_response['p_val'] < p_threshold) & (pvalues_response['maxdeltaF'] >= maxdeltaFrequired)]
        selected_rois = significantpval['roi'].unique()
    else:
        raise ValueError("Invalid inclusion criterion. Choose from 'IC1', 'IC2', 'IC3', 'IC4', 'IC5', 'IC6', or 'IC7'.")
    
    # Prepare time axis
    time = np.linspace(-secondsbefore, secondsafter, windows.shape[2])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Plot population response lines
    for cond, color in zip(condition_columns, condition_colors):
        responsive_data = [resp for roi, resp in zip(range(len(roi_columns)), all_responses[cond]) if roi in selected_rois]
        if responsive_data:
            mean_response = np.mean(np.mean(responsive_data, axis=0), axis=0)
            sem_response = np.std(np.mean(responsive_data, axis=0), axis=0) / np.sqrt(len(responsive_data))
            
            ax1.plot(time, mean_response, color=color, label=cond)
            ax1.fill_between(time, mean_response-sem_response, mean_response+sem_response, color=color, alpha=0.2)

    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Mean ΔF/F%')
    ax1.set_title(f'Population Response of Stimuli-Responsive ROIs\n{inclusion_criterion} (n={len(selected_rois)})')
    ax1.legend()


    #####################
    
    # Prepare data for condition heatmap
    heatmap_data = []
    for cond in condition_columns:
        responsive_data = [resp for roi, resp in zip(range(len(roi_columns)), all_responses[cond]) if roi in selected_rois]
        if responsive_data:
            mean_response = np.mean(np.mean(responsive_data, axis=0), axis=0)
            heatmap_data.append(mean_response)
    
    heatmap_data = np.array(heatmap_data)
    
    # Plot condition heatmap
    sns.heatmap(heatmap_data, ax=ax2, cmap='viridis', cbar_kws={'label': 'Mean ΔF/F%'})
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Orientation')
    ax2.set_title('Stimuli Response Heatmap')
    ax2.set_xticks(np.linspace(0, len(time)-1, 5))
    ax2.set_xticklabels([f'{t:.1f}' for t in np.linspace(time[0], time[-1], 5)], ha='center', rotation=0)
    ax2.set_yticks(np.arange(len(condition_columns)) + 0.5)
    ax2.set_yticklabels(condition_columns, rotation=0, ha='right')
    
    # Add vertical line at time 0
    zero_index = np.argmin(np.abs(time))  # Find the index closest to time 0
    ax2.axvline(x=zero_index, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of responsive ROIs: {len(selected_rois)}")
    print(f"Responsive ROIs: {sorted(selected_rois)}")
    
   # Prepare data for averaged ROI heatmap
    roi_heatmap_data = []
    for roi in selected_rois:
        roi_responses = []
        for cond in condition_columns:
            roi_responses.extend(all_responses[cond][roi])
        mean_response = np.mean(roi_responses, axis=0)
        roi_heatmap_data.append(mean_response)
    
    roi_heatmap_data = np.array(roi_heatmap_data)
    
    # Scale the data to 0-100 range
    min_val = np.min(roi_heatmap_data)
    max_val = np.max(roi_heatmap_data)
    roi_heatmap_data_scaled = (roi_heatmap_data - min_val) / (max_val - min_val) * 100
    
    if reorder_rois:
        # Find the time of peak response after stimulus onset for each ROI
        stimulus_onset_idx = np.argmin(np.abs(time))  # Find the index closest to time 0
        peak_times = np.argmax(roi_heatmap_data_scaled[:, stimulus_onset_idx:], axis=1) + stimulus_onset_idx
        
        # Sort ROIs based on their peak response time (in reverse order)
        sorted_indices = np.argsort(peak_times)[::-1]  # Reverse the order
        roi_heatmap_data_sorted = roi_heatmap_data_scaled[sorted_indices]
        sorted_roi_labels = [f'ROI {selected_rois[i]}' for i in sorted_indices]
    else:
        # Keep original order
        roi_heatmap_data_sorted = roi_heatmap_data_scaled
        sorted_roi_labels = [f'ROI {roi}' for roi in selected_rois]
    
    # Create compact Plotly heatmap
    plotly_fig = go.Figure(data=go.Heatmap(
        z=roi_heatmap_data_sorted,
        x=time,
        y=sorted_roi_labels,
        colorscale='Viridis',
        zmin=0,
        zmax=100,
        colorbar=dict(title='%ΔF/F%', titleside='right', thickness=15, len=0.9)
    ))
    
    plotly_fig.update_layout(
        title=dict(
            text=f'ROI Responses ({inclusion_criterion}, n={len(selected_rois)}{"" if reorder_rois else ""})',
            font=dict(size=14),
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Time (s)',
        yaxis_title='ROI' + (' (sorted by peak time, latest to earliest)' if reorder_rois else ''),
        width=800,
        height=600,
        margin=dict(l=50, r=50, t=30, b=30),
        xaxis=dict(tickmode='array', tickvals=np.arange(-secondsbefore, secondsafter+1)),
        yaxis=dict(tickmode='array', tickvals=np.arange(0, len(selected_rois), max(1, len(selected_rois)//10))),
    )
    
    # Add vertical line at stimulus onset
    plotly_fig.add_shape(
        type="line",
        x0=0, y0=0, x1=0, y1=1,
        yref="paper",
        line=dict(color="white", width=2)
    )
    
    plotly_fig.show()
    
    print(f"Number of responsive ROIs: {len(selected_rois)}")
    if reorder_rois:
        print(f"Responsive ROIs (sorted by peak time, latest to earliest): {[selected_rois[i] for i in sorted_indices]}")
    else:
        print(f"Responsive ROIs (original order): {list(selected_rois)}")
    print(f"Original ΔF/F% range: {min_val:.4f} to {max_val:.4f}")

    # Create DataFrame with responsive neurons and non-ROI columns
    selected_roi_columns = [roi_columns[i] for i in selected_rois]
    responsive_df = traces_orig[selected_roi_columns + non_roi_columns]
    return responsive_df, plotly_fig