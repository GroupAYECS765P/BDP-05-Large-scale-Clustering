from pyspark import SparkContext, SparkConf
import re
import numpy as np
import copy as cp

# The K Means Methods
langSpread = 50000 # K-means parameter: How 'far apart' languages should be for the kmeans algorithm?
kmeansKernels = 45 # K-means parameter: Number of clusters
kmeansEta = 20.0 # K-means parameter: Convergence criteria
kmeansMaxIterations = 120 # K-means parameter: Maximum iterations

# Languages Dictionary
langDict = {'javascript':0 ,'java':1, 'php':2,'python':3,'c#':4,'c++':5,'ruby':6,'css':7,'objective-c':8,
            'perl':9,'scala':10,'haskell':11,'matlab':12,'clojure':13,'groovy':14, 'html':15, 'asp.net':17}

# Sample the vectors
def sampleVectors(vectors):
    return vectors.takeSample(False, kmeansKernels, kmeansKernels)

# Main K Means Computation
def kmeans(means, vectors, iter=1, debug=False):
    # Map argument vectors and obtain the closest centroid.
    # group all points by its closest centroid
    # Map each cluster and find the new centroid by averaging all points.
    # Update the array means.

    newMeans = cp.copy(means)
    centroids = vectors.map(lambda v: (findCloset(v, means), v)).groupByKey().mapValues(lambda vs: averageVectors(vs)).collect()
    for i, centroid in centroids:
        newMeans[i] = centroid

    distance = euclideanDistanceArray(means, newMeans)
    if converged(distance):
        return newMeans
    elif iter < kmeansMaxIterations:
        kmeans(newMeans, vectors, iter + 1, debug)
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
def findCloset(p, centers):
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

# Displaying results
def clusterResults(means, vectors):
    closest = vectors.map(lambda p: (findClosest(p, means), p))
    closestGrouped = closest.groupByKey()
    median = closestGrouped.mapValues(lambda vs: calMedian(vs))
    return median.collect().map(vs[1]).sortBy(vs[3])

def calMedian(vs):
    # most common language in the cluster
    langMode = vs.groupBy(vs[0]).mapValues(len(vs)).maxBy(vs[1])[0]
    langLabel = langs[langMode / langSpread]
    clusterSize = len(vs)
    # percent of the questions in the most common language
    langPercent = len(vs.filter(vs[0] == langMode)) / clusterSize * 100
    scores = np.sort(np.array(vs.unzip[1]))
    medianScore = calMedianScore(scores)
    return (langLabel, langPercent, clusterSize, medianScore)

def calMedianScore(scores):
    mid = len(scores) / 2
    if (len(scores) % 2 == 0):
        return (scores[mid] + scores[mid - 1]) / 2
    else:
        return scores(mid)

def printResults(results):
    print('Resulting clusters:')
    print('Score, Dominant language, percent, Questions')
    print('================================================')
    for (lang, percent, size, score) in results:
        print('{}, {}, {}, {}'.format(lang, percent, size, score))

# Main Function
conf = SparkConf().setAppName('Large-Scale Clustering').set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf = conf)

lines = sc.textFile('coursework2/kmeansVec.txt')
vectors = lines.map(lambda line: np.array([float(x) for x in line.split(',')]))

means = kmeans(sampleVectors(vectors), vectors, True)
inmem = means.persist()
inmem.saveAsTextFile('coursework2/means')
results = clusterResults(means, vectors)
printResults(results)
sc.stop()
