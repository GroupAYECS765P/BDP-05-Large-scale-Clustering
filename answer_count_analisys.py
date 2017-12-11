###############################################################################
# This code aims read the stackOverFlow dataset and extract reputation and total
# amount of answers provided for all the users.
# And submit this info through K-Means clustering
###############################################################################
import re
import xml.etree.ElementTree as ET
from numpy import array
from math import sqrt
import random

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

CLUSTERS_NUMBER = 4

conf = SparkConf().setAppName("Large-Scale Clustering")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)

def is_user( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        return 'Reputation' in xmlObj.attrib and 'Id' in xmlObj.attrib
    except:
        return False

def is_valid_answer( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        if 'PostTypeId' in xmlObj.attrib and 'OwnerUserId' in xmlObj.attrib and 'Score' in xmlObj.attrib:
            return xmlObj.attrib['PostTypeId'] == '2'
        else:
            return False
    except:
        return False


def get_reputation( xmlStr ):
    xmlObj = ET.fromstring(xmlStr)
    return (xmlObj.attrib['Id'], int(xmlObj.attrib['Reputation']))


def get_user_answer(xmlStr):
    xmlObj = ET.fromstring(xmlStr)
    return (xmlObj.attrib['OwnerUserId'], 1)

records = sc.textFile("/data/stackOverflow2017")

# FILTERING STAGE
users = records.filter(is_user)
answers = records.filter(is_valid_answer)

# MAPPING STAGEt
user_vector = users.map(get_reputation)
answer_count_vector = answers.map(get_user_answer).reduceByKey(lambda a, b : a+b)

# INNER JOIN OF THE RDDS
kmeans_vector = answer_count_vector.join(user_vector).persist()

# K-MEANS STAGE
clusters = KMeans.train(kmeans_vector.values(), CLUSTERS_NUMBER, maxIterations=10, initializationMode="random")

# PLOTTING ORGANIZATION STAGE
def get_plot_data ( input_data , kmeans_model):
    result = kmeans_model.predict(input_data)
    return [input_data, result]

plot_data = kmeans_vector.values().map(lambda data : get_plot_data(data, clusters))

plot_data.saveAsTextFile('dataPlot')