Link to download the Kmeans Vectors Folder to test the Kmeans Algorithm:

https://goo.gl/iTUyrM

KmeansVector file contains the output vectors from Spark

KmeansVec file contains a more readble version since you can slipt just by ','

plotData file contains a clusterised data (using MLlib) with 40 clusters and 10 max iterations, the third columm is the cluster of the data in the same line.
___________________________________________

Link to edit the .tex file with the report: https://pt.sharelatex.com/1335567134sxggmcgtcwzv

To run scripts in python using pyspark use: spark-submit name_of_file.py

To understand the database: https://ia800500.us.archive.org/22/items/stackexchange/readme.txt

# BDP 05: CLUSTERING OF LARGE UNLABELED DATASETS
Overview
=========
Real world data is frequently unlabeled and can seem completely random. In these sort of situations, unsupervised learning techniques are a great way to find underlying patterns. This project looks at one such algorithm, KMeans clustering, which searches for boundaries separating groups of points based on their differences in some features.

The goal of the project is to implement an unsupervised clustering algorithm using a distributed computing platform. You will implement this algorithm on the stack overflow user base to find different ways the community can be divided, and investigate what causes these groupings.

The clustering algorithm must be designed in a way that is appropriate for data intensive parallel computing frameworks. Spark would be the primary choice for this project, but it could also be implemented in Hadoop MapReduce. Algorithm implementations from external libraries such as Spark MLib may not be utilised; the code must be original from the students. However, once the algorithm is completed, a comparison between your own results and that generated by MLlib could be interesting and aid your investigation.

Stack Overflow is the main dataset for this project, but alternative datasets can be adopted after consultation with the module organiser. Additionally, different clustering algorithms may be utilised, but this must be discussed and approved y the module organiser. 

DATASET
=========
The project will use the Stack Overflow dataset. This dataset is located in HDFS at /data/stackoverflow
The dataset for StackOverflow is a set of files containing Posts, Users, Votes, Comments, PostHistory and PostLinks. Each file contains one XML record per line.
For complete schema information: Click here

In order to define the clustering use case, you must define what should be the features of each post that will be used to cluster the data. Have a look at the different fields to define your use case. 

# ALGORITHM
The project will implement the k-means algorithm for clustering. This algorithm iteratively recomputes the location of k centroids (k is the number of clusters, defined beforehand), that aim to classify the data. Points are labelled to the closest centroid, with each iteration updating the centroids location based on all the points labelled with that value.

Spark and Map/Reduce can be utilised for implementing this problem. Spark is recommended for this task, due to its performance benefits in . However, note that the MLib extension of Spark is not allowed to be used as the primary implementation.  The group must code its own original implementation of the algorithm. However, it is possible to also use the mllib implementation, in order to evaluate the results from  each clustering implementation.

Report Contents
=========
Brief literature survey on clustering algorithms, including the challenges on implementing them at scale for parallel frameworks. The report should then justify the chosen algorithm (if changed) and the implementation.

Definition of the project use case, where the implemented project will be part of the solution.

Implementation in MapReduce or Spark of a clustering algorithm(KMeans). Must take into account the potential enormous size of the dataset, and develop sensible code that will scale and efficiently use additional computing nodes. The code will also need to potentially convert the dataset from its storage format to an in-memory representation. Source code should not be included in the report. However, the algorithms should be explained in the report.

Results section. Adequate figures and tables should be used to present the results. The effectiveness of the algorithm should also be shown, including performance indications. Not really sure if this can be done for clustering. Critical evaluation of the results should be provided.

Experiments demonstrating the technique can successfully group users in the dataset. Representation of the results, and discussion of the findings in a critical manner. 

ASSESSMENT
=========
The project according to the specification has a base difficulty of 85/100. This means that a perfect implementation and report would get a 85. Additional technical features and experimentation would raise the difficulty in order to opt for a full 100/100 mark.

Report presentation: 20%
Appropriate motivation for the work. Lack of typos/grammar errors, adequate format. Clear flow and style. Related work section including  adequate referencing. 

Technical merit: 50%
Completeness of the implementation. [25%]
Provided source code. Code is documented. [10%]
Design rationale of the code is provided. [10%]
Efficient, and appropriate implementation for the chosen platform. [5%]

Results/Analysis: 30%
Experiments have been carried out on the full dataset. [10%] 
Adequate plots/tables are provided, with captions. [10%] 
Results are not only presented but discussed appropriately. [10%]

Additional project goals: 
Implementation of additional functions beyond the base specification can raise the base mark up to 100. A non-exhaustive list of expansion ideas include:
Exploration and discussion of hyperparameter tuning (e.g. the number of k groups to cluster the data into) [up to 10 marks]
Comparative evaluation of clustering technique with existing implementations (e.g. mllib) [up to 10 marks]
Bringing in additional datasets from stackoverflow, such as user badges, to aid in clustering [up to 5 marks]
Cluster additional datasets (such as posts) [up to 10 marks]

LEAD DEMONSTRATOR
=========
For specific queries related to this coursework topic, please liaise with Mr/Ms TBD, who will be the lead demonstrator for this project, as well as with the module organiser.

SUBMISSION GUIDELINES
=========
The report will have a maximum length of 8 pages, not counting cover page and table of contents.
The report must include motivation of the problem, brief literature survey, explanation of the selected technique, implementation details and discussion of the obtained results, and references used in the work.
Additionally, the source code must be included as a separate compressed file in the submission.
