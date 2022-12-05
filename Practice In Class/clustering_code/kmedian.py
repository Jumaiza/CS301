
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
#######sklearn dont hvae kmedian package hence we utilse pyclustering
# Load list of points for cluster analysis.
#########simple data with two coloumns each represents two cluster values
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
########we pass random data points as initial median
# Create instance of K-Medians algorithm.
initial_medians = [[0.0, 0.1], [2.5, 0.7]]
kmedians_instance = kmedians(sample, initial_medians)

# Run cluster analysis and obtain results.
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = kmedians_instance.get_medians()
########to see the generated cluster and median we append those values to the cluster visualiser
# Visualize clustering results.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(initial_medians, marker='*', markersize=10)
visualizer.append_cluster(medians, marker='*', markersize=10)
visualizer.show()

