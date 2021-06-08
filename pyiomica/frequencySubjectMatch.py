# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from collections import Counter
from copy import deepcopy

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


def bootstrapGeneral(df, N, shuffling = True):
    '''To generate bootstrap samples
    
    Parameters:
        df: pandas dataframe
            the source dataframe using to generate bootstrap samples
        N: integer
            the size of bootstrap samples
        shuffling : boolean
            shuffle the data or not, The default is True.
    
    Returns:
        bootstrapDF: pandas dataframe
            the bootstrap samples
    '''


    bootstrapDF = pd.DataFrame()
    df.reset_index(drop=True, inplace=True)
    if shuffling:
        for column in df:
            bootstrapDF[column] = df[column].iloc[np.random.randint(df.shape[0], size=N)].values
            
    else:
        bootstrapDF = df.iloc[np.random.randint(df.shape[0], size=N)]
    
    return bootstrapDF

def calculateLinksBetweenSubjectsByDistance(df1,df2,cutoff):
    '''To calculate the linked time series/Genes from two dataframes base on the Euclidean distance
    
    Parameters:
        df1: pandas dataframe
            the first time series from df1
        df2: pandas dataframe
            the second time series from df2
        cutoff: float
            if the distance between two time series less than cutoff, the two time series is linked
            time series
    
    Returns:
        numlinkedGenes: integer/float
            number of linked time series
        commonGenes: integer/float
            number of common time series in df1 and df2
        linkedGenes: list of string
            the ids/names of linked time series
    '''
    
    idx = df1.index.intersection(df2.index)
    numlinkedGenes = 0
    linkedGenes = []
    commonGenes = len(idx)
    if commonGenes != 0:
        for r in idx:
            s1 = df1.loc[r]
            s2 = df2.loc[r]
            
            d = np.linalg.norm(s1-s2)
            if d < cutoff:
                numlinkedGenes += 1
                linkedGenes.append(r)
    return numlinkedGenes, commonGenes,linkedGenes

def calculateLinksBetweenSubjectsByCorrelation(df1,df2,cutoff):
    '''To calculate the linked time series/Genes from two dataframes base on the pearson correlation
    
    Parameters:
        df1: pandas dataframe
            the first time series from df1
        df2: pandas dataframe
            the second time series from df2
        cutoff: float
            if the pearson correlation between two time series less than cutoff, the two time series is linked
            time series
    
    Returns:
        numlinkedGenes: integer/float
            number of linked time series
        commonGenes: integer/float
            number of common time series in df1 and df2
        linkedGenes: list of string
            the ids/names of linked time series
    
    '''
    
    idx = df1.index.intersection(df2.index)
    numlinkedGenes = 0
    linkedGenes = []
    commonGenes = len(idx)
    if commonGenes != 0:
        for r in idx:
            s1 = df1.loc[r]
            s2 = df2.loc[r]
            
            d = pearsonr(s1, s2)[0]
            if d > cutoff:
                numlinkedGenes += 1
                linkedGenes.append(r)
    return numlinkedGenes, commonGenes,linkedGenes

def getCommunityStructure(cs):
    '''
    To change community structure from {node1:community1, node2:community2,...} to
    {community1:[node1, node2,...], community2:[node3, node4,...]}

    Parameters:
        cs: dictionary
            the community structure as {node1:community1, node2:community2,...}
        
    Returns:
        community_structure: dictionary
            the community structure as {community1:[node1, node2,...], community2:[node3, node4,...]}
    '''
    commu_list = list(cs.values())
    community_structure = {}
    for commu in commu_list:
        templist = []
        for k,v in cs.items():
            if int(v) == int(commu):
                templist.append(k)
        community_structure[commu] = deepcopy(templist)
    return community_structure

def getCommunityGenesDict(community_structure,genelist,endwithString):
    '''To get gene IDs list of each community within selected individuals' category 

    Parameters:
        community_structure: dictionary
                the community structure as {community1:[node1, node2,...], community2:[node3, node4,...]}
        genelist: dictionary
            the gene list of each individuals, the key is the id of individual 
        endwithString: list of string
           the selected individuals categories, which attached to the end of the individual ids 
    
    Returns:
        community_genes_dict: dictionary
            the genes list of each community
    
    '''

    community_genes_dict = {}
    for key in community_structure.keys():
    #for k in [1]:    
        genes_list = []
        subjectlist = community_structure[key]
        for k in genelist.keys():
            if (k[0].endswith(endwithString) and k[1].endswith(endwithString)
                and k[0] in subjectlist and k[1] in subjectlist):
            #if sub.endswith('Prediabetic'):
                genes_list.append(set(genelist[k]))
            
        #copy_genes_list = deepcopy(genes_list)
        
        if len(genes_list) > 0:
            genes_set = list(set.intersection(*genes_list))
            if len(genes_set):
                community_genes_dict[key] = genes_set
    community_genes_dict = {k: v for k, v in community_genes_dict.items() if len(v)}
    return community_genes_dict

def splitGenes(community_gene_dict):
    '''Split gene ids, to seperate the genes name from attached labels

    Parameters:
        community_gene_dict: dictionary
            the genes ids list of each community
    
    Returns:
        new_dict: dictionary
            the gene names list of each community
    '''
    new_dict = {}
    for key in community_gene_dict:
        fl = []
        l = community_gene_dict[key]
        for name in l:
            fl.append(name.split('+')[0])
        new_dict[key] = fl
        
    return new_dict

def getCommunityTopGenesByNumber(community_structure,genelist,endwithString,numberOfTopGenes=500):
    '''To get the top ranking genes of each community

    Parameters:
        community_structure: dictionary
            the community structure as {community1:[node1, node2,...], community2:[node3, node4,...]}
        genelist: dictionary
            the genes list of each community 
        endwithString: list of string
            the selected individuals categories, which attached to the end of the individual ids 
        numberOfTopGenes: integer, optional
            the number of top ranking genes. The default is 500.
    
    Returns:
        community_genes_dict: dictionary
            the top ranking genes of each community
    '''

    community_genes_dict = {}
    for key in community_structure.keys():
    #for k in [1]:    
        genes_list = []
        subjectlist = community_structure[key]
        for k in genelist.keys():
            if (k[0].endswith(endwithString) and k[1].endswith(endwithString)
                and k[0] in subjectlist and k[1] in subjectlist):
            #if sub.endswith('Prediabetic'):
                genes_list.extend(genelist[k])
            
        counted = Counter(genes_list)
        ordered = [value for value, count in counted.most_common()]
        
        if len(ordered) >= numberOfTopGenes:
            community_genes_dict[key] = ordered[0:numberOfTopGenes]
        else:
            print("number of genes less than choose number of top genes")
            community_genes_dict[key] = ordered
            
    community_genes_dict = {k: v for k, v in community_genes_dict.items() if len(v)}
            
    return community_genes_dict

def getCommunityTopGenesByFrequencyRanking(community_structure,genelist,endwithString,frequencyPercentage=50):
    '''To get the top frequency genes of each community
    
    Parameters:
        community_structure: dictionary
            the community structure as {community1:[node1, node2,...], community2:[node3, node4,...]}
        genelist: dictionary
            the genes list of each community 
        endwithString: list of string
            the selected individuals categories, which attached to the end of the individual ids 
        frequencyPercentage: float, optional
            the top percentage frequency of choosed genes, The default is 50.
    
    Returns:
        community_genes_dict: dictionary
            the top percentage frequency genes of each community
    '''

    community_genes_dict = {}
    for key in community_structure.keys():
    #for k in [1]:    
        genes_list = []
        subjectlist = community_structure[key]
        for k in genelist.keys():
            if (k[0].endswith(endwithString) and k[1].endswith(endwithString)
                and k[0] in subjectlist and k[1] in subjectlist):
            #if sub.endswith('Prediabetic'):
                genes_list.extend(genelist[k])
            
        counted = Counter(genes_list)
        top_ranking_value = round(max(counted.values()) * (1 - frequencyPercentage/100))
        genes = [k for k, v in counted.items() if int(v) >= top_ranking_value]

        community_genes_dict[key] = genes
           
    community_genes_dict = {k: v for k, v in community_genes_dict.items() if len(v)}
            
    return community_genes_dict

def optimizeK(df,rangeK,saveFig=False,**kargs):
    '''To optimize the k value of k-mean cluster

    Parameters:
        df: pandas dataframe
            the data source to do k-mean cluster
        rangeK: python range, e.g. rangeK = range(0,10)
            the K value range
        saveFig: boolean, optional
            save figure or not. The default is False.
        \*\*kargs: figure name
            if saveFig is true, the \*\*kargs is the figure name 
    
    Returns:
        optimizek:integer
            the optimized K value
    
    '''
    
    sil = []
    elbow = []
    K = rangeK
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(df)
        labels = km.labels_
        elbow.append(km.inertia_)
        sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    
    maxindex = sil.index(max(sil))
    optimizek = K[maxindex]
    
    if saveFig:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6)) # create figure & 1 axis
        ax1.plot(K, sil, 'bx-')
        ax1.set_xlabel('k')
        ax1.set_ylabel('Silhouette_score')
        ax1.set_title('Silhouette Method For Optimal k')
        
        ax2.plot(K, elbow, 'rx-')
        ax2.set_xlabel('k')
        ax2.set_ylabel('Sum_of_squared_distances')
        ax2.set_title('Elbow Method For Optimal k')
        plt.tight_layout()
        fig.savefig(kargs['figname'])   # save the figure to file
        plt.close(fig) 
    
    return optimizek
    