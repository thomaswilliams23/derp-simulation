import h5py
import numpy as np
import scipy.stats as stats

#look at an hdf5 file, pull out present prevalence and cumulative incidence



#DB_FNAME = "out/sim-wartortle/dataset-wartortle.hdf5"
DB_FNAME = "out/sim-charmeleon-alt/dataset-charmeleon-alt.hdf5"

print(f"\nReading from {DB_FNAME}\n")


#read in
db_conn = h5py.File(DB_FNAME, "r")

present_times = np.array([db_conn[f"{key}/input/present"][()] for key in db_conn.keys()])

r0_values = np.array([db_conn[f"{key}/output/parameters/r0/values"][()] for key in db_conn.keys()])
r0_ct = np.array([db_conn[f"{key}/output/parameters/r0/change_times"][()] for key in db_conn.keys()])

net_removal_rates = np.array([db_conn[f"{key}/output/parameters/net_removal_rate/values"][()] for key in db_conn.keys()])
net_removal_ct = np.array([db_conn[f"{key}/output/parameters/net_removal_rate/change_times"][()] for key in db_conn.keys()])

present_prevalence = np.array([db_conn[f"{key}/output/present_prevalence"][()] for key in db_conn.keys()])
present_cumulative = np.array([db_conn[f"{key}/output/present_cumulative"][()] for key in db_conn.keys()])





#







#OUTPUT

#prevalence
prevalence_mean = np.mean(present_prevalence)
prevalence_median = np.median(present_prevalence)
prevalence_ci = stats.t.interval(
    0.95, 
    len(present_prevalence)-1, 
    loc=prevalence_mean, 
    scale=stats.sem(present_prevalence)
)
print(f"Present Prevalence: Median={prevalence_median:.4f}, 95% CI=({prevalence_ci[0]:.4f}, {prevalence_ci[1]:.4f})")


#cumulative incidence
cumulative_mean = np.mean(present_cumulative)
cumulative_median = np.median(present_cumulative)
cumulative_ci = stats.t.interval(
    0.95, 
    len(present_cumulative)-1, 
    loc=cumulative_mean, 
    scale=stats.sem(present_cumulative)
)
cumulative_median = np.median(present_cumulative)
print(f"Present Cumulative Incidence: Median={cumulative_median:.4f}, 95% CI=({cumulative_ci[0]:.4f}, {cumulative_ci[1]:.4f})")