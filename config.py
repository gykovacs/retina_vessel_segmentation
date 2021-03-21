import os

output_dir= 'output'
figures_dir= os.path.join(output_dir, 'figures')
latex_dir= os.path.join(output_dir, 'latex')
drive_dir= os.path.join('data', 'drive')

image_stats_file= os.path.join(output_dir, 'drive_stats.csv')
xls_file= os.path.join('data', 'retina_vessel_segmentation_final.xlsx')
image_level_results_file= os.path.join(output_dir, 'results_with_image_level_data.csv')
aggregated_results_file= os.path.join(output_dir, 'results_with_aggregated_data.csv')
adjusted_scores_file= os.path.join(output_dir, 'results_with_adjusted_scores.csv')

image_level_threshold= 0.5
aggregated_threshold= 0.5

exclude_stare_training= True
