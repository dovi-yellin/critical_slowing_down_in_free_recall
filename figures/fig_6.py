import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


full_filename = r'data\fig6_model_network_size_100_plus_gamma_0.08_plus_rep_5.csv'
df = pd.read_csv(full_filename)
mean_pwr_arr_df = df[['gamma','network_size','norm_sum_power']].groupby(['gamma','network_size'])['norm_sum_power'].mean().to_frame().reset_index()

mean_pwr_arr_flat = mean_pwr_arr_df['norm_sum_power'].values
mean_pwr_arr = mean_pwr_arr_flat.reshape(-1,11,10)

mean_pwr_arr = np.squeeze(mean_pwr_arr, axis=0)
mean_pwr_arr_ud = np.flipud(mean_pwr_arr)

mean_pwr_arr_base_norm = (mean_pwr_arr_ud / mean_pwr_arr_ud[-1,0])
mean_pwr_arr_base_norm_log = np.log10(mean_pwr_arr_base_norm)
mean_pwr_arr_column_norm = (mean_pwr_arr_ud.T / mean_pwr_arr_ud[:,0]).T
mean_pwr_arr_row_mean = (mean_pwr_arr_ud.T / np.mean(mean_pwr_arr_ud, axis=1)).T

font_size = 25
fig,ax = plt.subplots(figsize=(12,12))
im = plt.imshow(mean_pwr_arr_ud, cmap='jet', extent=(-3, 3, 3, -3), interpolation='bilinear')
ax.set_xticks([-3, 0, 3])
plt.xticks(fontsize=font_size)
ax.tick_params(axis='x', pad=20)
ax.set_xticklabels(['150','250','350'])
ax.set_yticks([-3, 0, 3])
plt.yticks(fontsize=font_size)
ax.set_yticklabels(['0.99','0.95','0.90'])
plt.xlabel('Network Size', fontsize=font_size*1.5)
plt.ylabel('Control Parameter', fontsize=font_size*1.5)
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=font_size)

plt.show()