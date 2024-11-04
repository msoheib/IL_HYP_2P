#####################################################
#Open deltaF and calculate sum for each angle on
#####################################################
import pandas as pd
import helper_functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

openpathtracesexp= 'aligned_dff30.csv'
column_names = helper_functions.generate_column_names(0, 92)
targetcolumns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315']
degreesarray=[0,45,90,135,180,225,270,315]
Select_columns_toplot=helper_functions.generate_column_names(87,92)

##########################################################################

#open and read dataframe with tracesRaw
deltaF=pd.read_csv(openpathtracesexp, delimiter=",", header=0, decimal='.',engine='python')
deltaF.set_index(['time'], inplace=True) 
print(deltaF)

angles=helper_functions.sum_columns_based_on_binary(deltaF,targetcolumns,column_names)
angles=angles.T
angles['degrees']=degreesarray

angles.to_csv('sumByangle.csv', sep=',', index = True, header=True)

#############################3
#PLOT 
helper_functions.plot_polar(angles,'degrees',Select_columns_toplot)
plt.show()

helper_functions.plot_polar_individual(angles,'degrees',Select_columns_toplot, commonmax=True)
plt.show()

figAngles=angles[Select_columns_toplot].plot(subplots=False, sharey=False, legend=True, colormap=cm.winter)
plt.show()

exit()
