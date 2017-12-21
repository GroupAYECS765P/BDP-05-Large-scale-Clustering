###############################################################################
# This code aims read the stackOverFlow dataset and extract reputation and total
# amount of answers provided for all the countries.
# And submit this info through K-Means clustering
###############################################################################
import re
import xml.etree.ElementTree as ET

from math import sqrt
import random

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

CLUSTERS_NUMBER = 4

conf = SparkConf().setAppName("Large-Scale Clustering")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)

def is_user_with_age( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        return 'Reputation' in xmlObj.attrib and 'Id' in xmlObj.attrib and 'Age' in xmlObj.attrib
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
    return (xmlObj.attrib['Age'], int(xmlObj.attrib['Reputation']))

def get_user_answer(xmlStr):
    xmlObj = ET.fromstring(xmlStr)
    return (xmlObj.attrib['OwnerUserId'], 1)

def get_age( xmlStr ):
    xmlObj = ET.fromstring(xmlStr)
    return (xmlObj.attrib['Id'], xmlObj.attrib['Age'])


users = sc.textFile("/data/stackOverflow2017/Users.xml")
posts = sc.textFile("/data/stackOverflow2017/Posts.xml")

# FILTERING STAGE
valid_users = users.filter(is_user_with_age)
answers = posts.filter(is_valid_answer)

# MAPPING STAGE
 #[userId, age]
user_age = valid_users.map(get_age)
#[age, reputation]
reputation_vector = valid_users.map(get_reputation)\
    .mapValues(lambda v: (v, 1))\
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))\
    .mapValues(lambda v: v[0]/v[1])
#[userId, answerCount]
answer_count_vector = answers.map(get_user_answer).reduceByKey(lambda a, b : a+b)
#[age, answerCount]
age_answer = user_age.join(answer_count_vector).map(lambda l : (l[1]))
age_answer = age_answer.mapValues(lambda v: (v, 1))\
    .reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))\
    .mapValues(lambda v: v[0]/v[1])

# INNER JOIN OF THE RDDS
data_vector = reputation_vector.join(age_answer) #(age, [mean reputation, mean answercount] )

# K-MEANS STAGE
clusters = KMeans.train(data_vector.values(), CLUSTERS_NUMBER, maxIterations=10, initializationMode="random")

# PLOTTING ORGANIZATION STAGE
def get_plot_data ( input_data , kmeans_model):
    result = kmeans_model.predict(input_data)
    return [input_data  , result]

plot_data = data_vector.map(lambda data : [data[0], get_plot_data(data[1], clusters)])

plot_data.saveAsTextFile('dataPlotAge')
