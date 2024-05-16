# publicRepo_Collinetal2024
code and processed data for (Collin et al., neural codes track prior events in a narrative and predict subsequent memory for details, on biorxiv: https://www.biorxiv.org/content/10.1101/2024.02.13.580092v2)

This repository contains all code used for (Collin et al., neural codes track prior events in a narrative and predict subsequent memory for details, on biorxiv: https://www.biorxiv.org/content/10.1101/2024.02.13.580092v2), commented but non-runnable, as well as processed data. The raw data can be found on openneuro (https://openneuro.org/datasets/ds005050). Below a description to explain what script has what purpose. 
behav_data_analyses/
This folder contains all code for analyzing the behavioral data (attention questions during video viewing for day 1 and 2), schema prediction questions during video viewing for day 1 and 2 (as used for figure 3A), pretraining task, schema test at end of day 2 (as used for figure 3B).
brainToBehav/brainToBehavLink.ipynb
This script calculates the within-subject brain (schema code, path code, rotated code, event code) to behavior (memory for details, memory for rituals) correlations and compares this to a null distribution (as used for table 1 and figure 9).
getSearchlightVectorRepresentation/run_job_tval.py
This script utilizes the information in the representational similarity analysis violin plots expanded at each TR to generate the vectors of t-values for each searchlight which are then fed into the K-Means algorithm.
kmeans/runClustering/tval/get_master_lists_tval.py
This script prepares us for the K-Means algorithm by gathering together the outputs of the run_job_tval.py script which outputs a file per searchlight containing the vectors of t-values.
kmeans/runClustering/tval/kmeans_clustering_tval.py
This script runs the K-Means clustering algorithm over searchlights where each searchlight is represented by a vector of t-values.
kmeans/run_visualization/tval/visualize_clustering_tval.py
 and kmeans/run_visualization/tval/visualize_clustering_tval_perS.py
These scripts prepare csv files that can be used to visualize the results of the K-Means algorithm by averaging together the data for all the searchlights within a particular cluster so we can get cluster-average representational similarity analysis plots. The perS indicates that the script allows for visualizations that zoom in on a particular participant’s representational similarity analysis comparison values for the searchlights within a particular cluster found by K-Means.
kmeans/silhoutte/tval/get_s_scores_sklearn_tval.py
This script computes the silhouette scores (figure A4, A16) for each searchlight from the K-Means results.
pvalues/pval_and_distill_part1/get_pvalue_each_searchlight.py
This script computes the ROI p-values for each searchlight which is the result of a t-test asking whether a particular pattern is present in this searchlight’s representational similarity analysis comparisons that is associated with the codes we were looking for a priori.
brainMaps/tval_evaluate_coloredBrain_fingerprints.ipynb
This script takes the calculated p-values for each searchlight as calculated before and plots that as a whole-brain .nii file (as used for figure 4). It saves both a brainmap with all p-values as well as an fdr-corrected one
rsa/get_distilled_neural_measures_SHPC.py
This script averages over searchlights to calculate average values for the brain to behavior link.
rsa/RSA_day2_2021march1_E2.py, RSA_day2_2021march1_E3.py and RSA_day2_2021march1_E4.py
These scripts are used to calculate the neural similarity timelines as used for the violinplots (described below). 
violinPlots/rsa/paperFigures.ipynb and paperFigures-GrandAverages.ipynb
These scripts take the rsa neural similarity TR-by-TR timelines, averages over time for each event, and create 3-by-3 plots (paperFigures.ipynb) or within vs across-stage averages (paperFigures-GrandAverages.ipynb), separately for each of othe 4 neural codes in the study (schema, rotated, path, event). The grandAverages script as used for figure 5, 6, 7, and 8. The paperFigures.ipynb as used for figure A1, A2, A3.
violinPlots/kmeans/paperFigures-kmeansFingerprints.ipynb and violinPlots/kmeans/paperFigures-kmeansFingerprints-filteredBySilhouette.ipynb
These scripts take the kmeans data to generate the 3-by-3 violin plots for each of the clusters of the K-means analysis, and for each of the K values tested (K = 2, 3, 4, 5). The script ending with -filteredBySilhouette creates the violin plot for the second cluster of K = 5 for which we ran a separate version on the top 25 % as defined by the silhouette scores. Code used for figure A5, A7, A10, A12, A14. 
violinPlots/kmeans/paperFigures-kmeansFingerprints-GrandAverages.ipynb and violinPlots/kmeans/paperFigures-kmeansFingerprints-GrandAverages-filteredBySilhouette.ipynb
These scripts take the K-means data to generate within-stage vs across-stage violin plots for each of the clusters in the K-means analysis, and for each of the K values tested (K = 2, 3, 4, 5). The script ending with -filteredBySilhouette creates the violin plot for the second cluster of K = 5 for which we ran a separate version on the top 25 % as defined by the silhouette scores. Code used for figure A6, A8, A9, A11, A13.
Figures 1, 2 and A15 are explanatory figures (not focused on data).
Processed data
data_for_all_behavTests
This folder has processed data csv files used for analyzing the attention questions during video viewing (day1_attentionQuestions.csv and day2_attentionQuestions.csv), schema learning during day 1 (day1_SchemaPrediction_coinTorchPrediction.csv and day1_SchemaPrediction_eggPaintingPrediction.csv), pre-training task (day2_pretraining_correctInFinalBlock.csv), day2 schema test (day2_test_for_schemalearning.csv), day2 stop-and-ask prediction questions (day2_SchemaPrediction.csv), and data of the recall task (day2_recall_EpisodicDetails_Correct.csv, day2_recall_EpisodicDetails_Incorrect.csv, day2_recall_Rituals_Correct.csv, day2_recall_Rituals_Incorrect.csv).
data_for_brainToBehaviorCorrelation/data
This folder has processed data used for the brain-to-behavior correlations (within subject) for each of the 4 neural codes assessed in the study.
data_for_brainToBehaviorCorrelation/nullDistributions
This folder has processed data used to test significance for the brain-to-behavior correlations (all correlations to memory for details as well as to memory for rituals was compared against a null distribution), for each of the 4 neural codes assessed in the study.
data_for_kmeans
This folder has processed data related to the Kmeans for K = 2, K = 3, K = 4, K = 5, K = 5 with filtering for the top25 percent, K = 5 cluster 5 when comparing template event 2 to event 2, K = 5 cluster 5 when comparing template event 2 to event 3, K = 5 cluster 5 when comparing template event 3 to event 2, K = 5 cluster 5 when comparing template event 3 to event 3.
data_for_rsa/brain_maps
This folder has processed data related to the RSA analyses in the form of nii files, one file for each of the 4 neural codes with all p-values throughout the brain and one file for each of the 4 neural codes with an fdr 0.05 threshold (saved inverted, i.e. 1 minus pvalue).
data_for_rsa/timelines
This folder has processed data concerning the rsa violin plots for each of the 4 neural codes assessed in the study. Each csv file has neural similarity values TR-by-TR for each participants. Each file used either event2, event3 or event4 template to compare it to, and either for other-event/other-schema, other-event/same-schema, same-event/same-schema or same-event/other-schema. 
