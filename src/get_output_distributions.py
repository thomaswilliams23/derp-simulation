import h5py
import numpy as np

#look at an hdf5 file, pull out present prevalence and cumulative incidence



DB_FNAME = "out/sim-wartortle/dataset-wartortle.hdf5"
#DB_FNAME = "out/sim-charmeleon-alt/dataset-charmeleon-alt.hdf5"

print(f"\nReading from {DB_FNAME}\n")


#read in
db_conn = h5py.File(DB_FNAME, "r")

epidemic_durations = np.array([db_conn[f"{key}/output/parameters/epidemic_duration"][()] for key in db_conn.keys()])

r0_values = [db_conn[f"{key}/output/parameters/r0/values"][()] for key in db_conn.keys()]
r0_ct = [db_conn[f"{key}/output/parameters/r0/change_times"][()] for key in db_conn.keys()]

net_removal_rates = [db_conn[f"{key}/output/parameters/net_removal_rate/values"][()] for key in db_conn.keys()]
net_removal_ct = [db_conn[f"{key}/output/parameters/net_removal_rate/change_times"][()] for key in db_conn.keys()]

sampling_prop_values = [db_conn[f"{key}/output/parameters/sampling_prop/values"][()] for key in db_conn.keys()]
sampling_prop_ct = [db_conn[f"{key}/output/parameters/sampling_prop/change_times"][()] for key in db_conn.keys()]

num_param_changes = np.array([len(r0_ct[ix]) for ix in range(len(r0_ct))])
all_change_times = np.concatenate(r0_ct)

present_prevalence = np.array([db_conn[f"{key}/output/present_prevalence"][()] for key in db_conn.keys()])
present_cumulative = np.array([db_conn[f"{key}/output/present_cumulative"][()] for key in db_conn.keys()])





#process parameter samples
all_r0_samples = np.concatenate(r0_values)
r0_weights = np.concatenate([
    np.diff(np.concatenate((np.array([0.0]), change_times, np.array([epidemic_durations[ix]])))) for (ix, change_times) in enumerate(r0_ct)
])

all_net_removal_samples = np.concatenate(net_removal_rates)
net_removal_weights = np.concatenate([
    np.diff(np.concatenate((np.array([0.0]), change_times, np.array([epidemic_durations[ix]])))) for (ix, change_times) in enumerate(net_removal_ct)
])

all_sampling_prop_samples = np.concatenate(sampling_prop_values)
sampling_prop_weights = np.concatenate([
    np.diff(np.concatenate((np.array([0.0]), change_times, np.array([epidemic_durations[ix]])))) for (ix, change_times) in enumerate(sampling_prop_ct)
])








# MODEL PARAMS
print("\nModel Parameter Distributions:")

#r0
r0_median = np.median(all_r0_samples)
r0_interval = np.percentile(all_r0_samples, [2.5, 97.5], weights=r0_weights, method='inverted_cdf')
print(f"R0: Median={r0_median:.4f}, 95% interval=({r0_interval[0]:.4f}, {r0_interval[1]:.4f})")

#net removal rate
net_removal_median = np.median(all_net_removal_samples)
net_removal_interval = np.percentile(all_net_removal_samples, [2.5, 97.5], weights=net_removal_weights, method='inverted_cdf')
print(f"Net Removal Rate: Median={net_removal_median:.4f}, 95% interval=({net_removal_interval[0]:.4f}, {net_removal_interval[1]:.4f})")

#sampling proportion
sampling_prop_median = np.median(all_sampling_prop_samples)
sampling_prop_interval = np.percentile(all_sampling_prop_samples, [2.5, 97.5], weights=sampling_prop_weights, method='inverted_cdf')
print(f"Sampling Proportion: Median={sampling_prop_median:.4f}, 95% interval=({sampling_prop_interval[0]:.4f}, {sampling_prop_interval[1]:.4f})")




#HYPERPARAMS
print("\nHyperparameter Distributions:")

#epidemic duration
epidemic_duration_median = np.median(epidemic_durations)
epidemic_duration_interval = np.percentile(epidemic_durations, [2.5, 97.5])
print(f"Epidemic Duration: Median={epidemic_duration_median:.4f}, 95% interval=({epidemic_duration_interval[0]:.4f}, {epidemic_duration_interval[1]:.4f})")

#number of parameter changes
num_param_changes_median = np.median(num_param_changes)
num_param_changes_interval = np.percentile(num_param_changes, [2.5, 97.5])
print(f"Number of Parameter Changes: Median={num_param_changes_median:.4f}, 95% interval=({num_param_changes_interval[0]:.4f}, {num_param_changes_interval[1]:.4f})")

#change times
all_change_times_median = np.median(all_change_times)
all_change_times_interval = np.percentile(all_change_times, [2.5, 97.5])
print(f"Change Times: Median={all_change_times_median:.4f}, 95% interval=({all_change_times_interval[0]:.4f}, {all_change_times_interval[1]:.4f})")




#OUTPUT
print("\nOutput Distributions:")

#prevalence
prevalence_median = np.median(present_prevalence)
prevalence_interval = np.percentile(present_prevalence, [2.5, 97.5])
print(f"Present Prevalence: Median={prevalence_median:.4f}, 95% interval=({prevalence_interval[0]:.4f}, {prevalence_interval[1]:.4f})")


#cumulative incidence
cumulative_median = np.median(present_cumulative)
cumulative_interval = np.percentile(present_cumulative, [2.5, 97.5])
print(f"Present Cumulative Incidence: Median={cumulative_median:.4f}, 95% interval=({cumulative_interval[0]:.4f}, {cumulative_interval[1]:.4f})")
print("\n")