from pyspark import SparkContext, SparkConf
import re
import numpy as np
import copy as cp
import itertools
import operator

# The K Means Methods
kmeansKernels = 4 # K-means parameter: Number of clusters
kmeansEta = 20.0 # K-means parameter: Convergence criteria
kmeansMaxIterations = 120 # K-means parameter: Maximum iterations

# Sample the vectors
def sampleVectors(vectors):
    return vectors.takeSample(False, kmeansKernels, 1)

# Main K Means Computation
def kmeans(means, vectors, iter=1):
    newMeans = cp.copy(means)
    centroids = vectors.map(lambda v: (findClosest(v, means), v)).groupByKey().mapValues(lambda vs: averageVectors(vs)).collect()
    for i, centroid in centroids:
        newMeans[i] = centroid
    distance = euclideanDistanceArray(means, newMeans)
    if converged(distance):
        return newMeans
    elif iter < kmeansMaxIterations:
        return kmeans(newMeans, vectors, iter + 1)
    else:
        print('Reached max iterations!')
        return newMeans

# Decide whether the kmeans clustering converged
def converged(distance):
    return distance < kmeansEta

def euclideanDistance(a1, a2):
    part1 = (a1[0]-a2[0])**2
    part2 = (a1[1]-a2[1])**2
    return part1 + part2

# Return the euclidean distance between two points
def euclideanDistanceArray(a1, a2):
    len1 = len(a1)
    len2 = len(a2)
    total = 0.0
    index = 0
    while (index < len1):
        total += euclideanDistance(a1[index], a2[index])
        index += 1
    return total

# Return the closest point
def findClosest(p, centers):
    bestIndex = 0
    closest = float('inf')
    for i in range(len(centers)):
        tempDist = euclideanDistance(p, centers[i])
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex

# Average the vectors
def averageVectors(ps):
    count = 0
    comp1 = 0
    comp2 = 0
    for p in ps:
        comp1 += p[0]
        comp2 += p[1]
        count += 1
    return (int(comp1/count), int(comp2/count))

# Utils for clustering results
def pyGroupByKey(keyIterList):
    it = itertools.groupby(keyIterList, operator.itemgetter(0))
    return [(key, sum(1 for item in subiter)) for key, subiter in it]

def pyMapValues(tupleList):
    dictMap = {}
    for k, v in tupleList:
        if dictMap.get(k):
            dictMap.update({k: dictMap.get(k) + v})
        else:
            dictMap.update({k: v})
    return dictMap

def pyMaxBy(dictMap):
    return max(dictMap.iterkeys(), key=(lambda key: dictMap.get(key)))

# Displaying results
def clusterResults(means, vectors):
    closest = vectors.map(lambda p: (findClosest(p[1], means), p)) # (clusters, (userId, (Key, Value)))
    inmem0 = closest.persist()
    inmem0.saveAsTextFile('coursework2/support_closest')
    closestGrouped = closest.groupByKey() # (clusters, List<(userId, (Key, Value))>)
    median = closestGrouped.mapValues(lambda vs: calMedian(vs))
    return sorted([v for k,v in median.collect()], key=lambda x: x[3])

def calMedian(vs): # (userId, (Key, Value))
    # most common language in the cluster
    vsv = (zip(*list(vs))[1])
    langMode = pyMaxBy(pyMapValues(pyGroupByKey(list(vsv))))  # {10: 1000, 20:100} -> 10
    clusterSize = float(len(list(vsv)))
    # percent of the questions in the most common language
    langPercent = len(filter(lambda arrPair: arrPair[0] == langMode, list(vsv))) / clusterSize * 100
    # scores = np.sort(np.array(zip(*list(zip(*list(list(vs)[1]))[1]))[1]))
    scores = np.sort(np.array(list(zip(*list(vsv))[1])))
    medianScore = calMedianScore(scores)
    return (langMode, langPercent, clusterSize, medianScore)

def calMedianScore(scores):
    mid = len(scores) / 2
    if (len(scores) % 2 == 0):
        return (scores[mid] + scores[mid - 1]) / 2
    else:
        return scores[mid]

def printResults(results):
    print('Resulting clusters:')
    print('    Score, Mode, Percent, Questions')
    print('================================================')
    for (mode, percent, size, score) in results:
        print('{:9}, {:4}, {:7.3f}, {:9}'.format(score, mode, percent, size))

# Main Function
conf = SparkConf().setAppName('Large-Scale Clustering').set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf = conf)

lines = sc.textFile('coursework2/support_vector.txt')
userIdVectors = lines.map(lambda line: eval(line))
vectors = lines.map(lambda line: np.array([float(x) for x in eval(line)[1]]))
vectors.persist()

means = kmeans(sampleVectors(vectors), vectors, 1)

inmem1 = sc.parallelize(means)
inmem1.saveAsTextFile('coursework2/support_means')

results = clusterResults(means, userIdVectors)
inmem2 = sc.parallelize(results)
inmem2.saveAsTextFile('coursework2/support_results')

printResults(results)

sc.stop()
