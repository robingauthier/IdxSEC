from scipy.cluster import hierarchy
import pandas as pd
import numpy as np
import os
import seaborn as sns

def hierarchical_clustering_wthreshold(distance_matrix, threshold=0.10, debug=True):
    """given a dataset we want to get clusters of columns correlated at more than 90%
    so that we just keep one of them.
    It will return the list of clusters and their members
    """

    linkage_matrix = hierarchy.linkage(distance_matrix, method="complete")  # You can choose a different linkage method
    clusters = hierarchy.fcluster(linkage_matrix, t=threshold, criterion="distance")
    variable_clusters = {variable: cluster for variable, cluster in zip(distance_matrix.columns, clusters)}

    clusdf = pd.Series(variable_clusters)\
        .to_frame('clusterid')\
        .sort_values('clusterid')\
        .reset_index()
    cluscnt = clusdf['clusterid']\
        .value_counts()\
        .to_frame('clustercnt')\
        .reset_index()\
        .rename(columns={'index': 'clusterid'})
    clusdf = clusdf.merge(cluscnt, on='clusterid', how='left')
    clusdf = clusdf.loc[lambda x: x['clustercnt'] > 1].copy()
    if debug:
        for clusterid in clusdf['clusterid'].unique().tolist():
            clusdfloc = clusdf[clusdf['clusterid'] == clusterid]
            print(clusdfloc)
    return clusdf.rename(columns={'index':'col'})

