import numpy as np
import json
import pandas as pd
import numpy as np
import os
import orjson

# this is to do the applied-to stuff2
# note that matching and applied-to are synonyms in my language
matching_id_labels_each_tr = []
for i in range(0,17):
    matching_id_labels_each_tr.append(0)
for i in range(17,23):
    matching_id_labels_each_tr.append(1)
for i in range(23,34):
    matching_id_labels_each_tr.append(2)
for i in range(34,50):
    matching_id_labels_each_tr.append(3)
for i in range(50,66):
    matching_id_labels_each_tr.append(4)
for i in range(66,74):
    matching_id_labels_each_tr.append(5)
matching_id_labels_each_tr = np.array(matching_id_labels_each_tr)


apply_id_to_it = {2: matching_id_labels_each_tr == 2,
                  3: matching_id_labels_each_tr == 3,
                  4: matching_id_labels_each_tr == 4}

path_names = ['sameEv-sameSchema', 'sameEv-otherSchema', 
                            'otherEv-sameSchema', 'otherEv-otherSchema']

path_abbrev_dict = {"sEsS": 'sameEv-sameSchema',
                    "sEoS": 'sameEv-otherSchema',
                    "oEsS": 'otherEv-sameSchema',
                    "oEoS": 'otherEv-otherSchema'}

roi_to_focus = {"schema": ["2_2", "3_2", "4_2", "2_3", "3_3", "4_3", "2_4", "3_4", "4_4"],
                "path": ["2_2", "3_2", "4_2", "2_3", "3_3", "4_3", "2_4", "3_4", "4_4"],
                "rotated": ["2_3", "3_2"],
                "perception": ["2_2","3_3","4_4"]}

roi_to_cluster_id = {"schema": 0,
                    "path":   1,
                    "rotated": 2,
                    "perception": 3}

job_id_to_roi = {0: "schema",
                 1: "path",
                 2: "rotated",
                 3: "perception"}

roi_new_names = ['schema', 'path','rotated', 'perception']

def get_cluster_id_to_distilled_searchlights():
    cluster_id_to_distilled_lights = {}
    for cluster_id in range(0,4):
        ranks_dir = "/scratch/gpfs/scollin/clustering_output/kmeans_searchlight_ranks/" + roi_new_names[cluster_id] + "_searchlight_list.csv"
        print(ranks_dir)
        ranks_df = pd.read_csv(ranks_dir)
        ranked_searchlights_in_cluster = ranks_df["searchlights_in_cluster"].tolist()
        cluster_id_to_distilled_lights[cluster_id] = ranked_searchlights_in_cluster

    return cluster_id_to_distilled_lights

# event should be of the form x_y where x is the template event and y is the applied-to event
job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
roi = job_id_to_roi[job_id]
cluster_id = roi_to_cluster_id[roi]
# get the searchlights in this cluster
cluster_id_to_distilled_lights = get_cluster_id_to_distilled_searchlights()
light_list = cluster_id_to_distilled_lights[cluster_id]
output_dir = "/scratch/gpfs/scollin/clustering_output/searchlights_distilled_neural_measures/roi_average/"
neural_measures_eachSearchlight_path = "/scratch/gpfs/scollin/clustering_output/searchlights_distilled_neural_measures/each_searchlight/"
pid_to_wedding_to_event_to_measures_allLights = {}
avg_pid_to_wedding_to_event_to_measure = {}
# aggregate across searchlights
for light_id in light_list:
    light_path = neural_measures_eachSearchlight_path + light_id
    with open(light_path, "rb") as f:
        pid_to_roi_to_wedding_to_event_to_measure = orjson.loads(f.read())
    for pid in pid_to_roi_to_wedding_to_event_to_measure:
        if pid not in pid_to_wedding_to_event_to_measures_allLights:
            pid_to_wedding_to_event_to_measures_allLights[pid] = {}
            avg_pid_to_wedding_to_event_to_measure[pid] = {}
        for wedding in pid_to_roi_to_wedding_to_event_to_measure[pid][roi]:
            if wedding not in pid_to_wedding_to_event_to_measures_allLights[pid]:
                pid_to_wedding_to_event_to_measures_allLights[pid][wedding] = {}
                avg_pid_to_wedding_to_event_to_measure[pid][wedding] = {}
            for event in pid_to_roi_to_wedding_to_event_to_measure[pid][roi][wedding]:
                if event not in pid_to_wedding_to_event_to_measures_allLights[pid][wedding]:
                    pid_to_wedding_to_event_to_measures_allLights[pid][wedding][event] = []
                measure = pid_to_roi_to_wedding_to_event_to_measure[pid][roi][wedding][event]
                # turn none into nan
                if measure == None:
                    measure = np.nan
                pid_to_wedding_to_event_to_measures_allLights[pid][wedding][event] += [measure]

# take average across searchlights
for pid in pid_to_wedding_to_event_to_measures_allLights:
    for wedding in pid_to_wedding_to_event_to_measures_allLights[pid]:
        for event in pid_to_wedding_to_event_to_measures_allLights[pid][wedding]:
            avg = np.nanmean(pid_to_wedding_to_event_to_measures_allLights[pid][wedding][event])
            avg_pid_to_wedding_to_event_to_measure[pid][wedding][event] = avg


# output the averages across searchlights
with open(output_dir + roi + "_per_ritual_within_wedding_notdistilled" +"_madeAfterPvalues.json", "w") as json_file:
    json.dump(avg_pid_to_wedding_to_event_to_measure, json_file)
