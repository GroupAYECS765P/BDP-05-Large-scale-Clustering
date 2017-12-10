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
    if 'Score' not in xmlObj.attrib:
        score = 0
    else:
        score = int(xmlObj.attrib['Score'])
    return (xmlObj.attrib['OwnerUserId'], score)

records = sc.textFile("/data/stackOverflow2017")
# records = sc.parallelize(['<row Id="138361019" PostId="26687223" VoteTypeId="2" CreationDate="2017-05-09T00:00:00.000" />', '<row Id="35391639" PostId="23145783" Score="0" Text="Sorry, I am still learning. I began learning 3 days ago... Still lots to learn. Thanks for the tip" CreationDate="2014-04-18T01:02:41.973" UserId="3546942" />', '<row Id="124925153" PostId="37706879" VoteTypeId="2" CreationDate="2016-10-25T00:00:00.000" />','<row Id="14351745" PostTypeId="1" AcceptedAnswerId="14354358" CreationDate="2013-01-16T05:08:36.767" Score="1" ViewCount="163" Body="&lt;p&gt;I\'m trying to figure out how to write a MySQL query that will return the nearest data which Actor = 210 for in terms E_id = 3.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;This is my original table:&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;Session              Player  E_id  Time                     Actor  PosX  PosY  index&#xA;-------------------  ------  ----  -----------------------  -----  ----  ----  -----&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    906   466   6&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    342   540   7&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  31     812   244   8&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   614   9&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  21     342   688   10&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  49     812   170   11&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  50     248   466   12&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    718   318   13&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  52     154   466   14&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  211    499   250   15&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   16&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   466   17&#xA;23131885ccc560bb6c8  10125   15    01-11-2012 08:56:38.323  20     718   318   18&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  52     154   466   19&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  11     499   250   20&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   21&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;If I fire query&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;select * from table where E_id = 3 or Actor = 210;&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;I get this result &lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;Session              Player  E_id  Time                     Actor  PosX  PosY  index&#xA;-------------------  ------  ----  -----------------------  -----  ----  ----  -----&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    906   466   6&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    342   540   7&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   614   9&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    718   318   13&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   16&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   466   17&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   21&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;&lt;strong&gt;expected Result is:&lt;/strong&gt;&#xA; row with index no 13 for row index no 16 and &#xA; row with index no 17 for row index no 21 &lt;/p&gt;&#xA;&#xA;&lt;p&gt;Index 16 and 21 both &lt;strong&gt;e_id&lt;/strong&gt; is &lt;strong&gt;3&lt;/strong&gt;&lt;/p&gt;&#xA;" OwnerUserId="1568236" LastEditorUserId="297408" LastEditDate="2013-01-16T09:40:25.020" LastActivityDate="2013-01-17T07:15:49.120" Title="SQL Query to show nearest data to specific event" Tags="&lt;php&gt;&lt;mysql&gt;&lt;sql&gt;" AnswerCount="4" CommentCount="11" />','<row Id="85641904" PostId="6518136" VoteTypeId="2" CreationDate="2015-03-21T00:00:00.000" />'])


# FILTERING STAGE
users = records.filter(is_user)
answers = records.filter(is_valid_answer)

# MAPPING STAGE
user_vector = users.map(get_reputation)
answer_count_vector = answers.map(get_user_answer).reduceByKey(lambda a, b : a+b)

# INNER JOIN OF THE RDDS
kmeans_vector = answer_count_vector.join(user_vector).persist()

# K-MEANS STAGE
clusters = KMeans.train(kmeans_vector.values(), CLUSTERS_NUMBER, maxIterations=10, initializationMode="random")

# PLOTTING STAGE
def get_plot_data ( input_data , kmeans_model):
    result = kmeans_model.predict(input_data)
    return [input_data, result]

plot_data = kmeans_vector.values().map(lambda data : get_plot_data(data, clusters))

plot_data.saveAsTextFile('dataPlot')