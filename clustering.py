import re
import xml.etree.ElementTree as ET
from numpy import array
from math import sqrt
import random

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

langDict = {'javascript': 0, 'java': 1, 'php': 2, 'python': 3, 'c#': 4, 'c++': 5, 'ruby': 6, 'css': 7, 'objective-c': 8, 'perl': 9, 'scala': 10, 'haskell': 11, 'matlab': 12, 'clojure': 13, 'groovy': 14, 'html': 15, 'r': 16, 'asp.net': 17}
LANG_DISTANCE = 100000
CLUSTERS_NUMBER = langDict.__len__()*3

conf = SparkConf().setAppName("Large-Scale Clustering")
sc = SparkContext(conf=conf)
sqlCtx = SQLContext(sc)

def hasTags( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        if 'Tags' in xmlObj.attrib:
            return True
    except:
        return False


def hasLangTags( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        if 'Tags' in xmlObj.attrib:
            tags = ET.fromstring(xmlStr).attrib['Tags']
            tags = re.split('>',tags)
            for tag in tags:
                tag = tag.replace('<', '')
                return tag in langDict
        else:
            return False
    except:
        return False


def is_valid_post_record( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        return ('ViewCount' in xmlObj.attrib or 'AnswerCount' in xmlObj.attrib or 'CommentCount' in xmlObj.attrib or 'FavoriteCount' in xmlObj.attrib) and 'Tags' in xmlObj.attrib
    except:
        return False


def is_valid_question_record( xmlStr ):
    try:
        xmlObj = ET.fromstring(xmlStr)
        return 'PostTypeId' in xmlObj.attrib and xmlObj.attrib['PostTypeId'] == '1' and 'Tags' in xmlObj.attrib
    except:
        return False


def getKmeans_activity_vector(xmlStr):
    xmlObj = ET.fromstring(xmlStr)
    if 'ViewCount' not in xmlObj.attrib:
        viewCount = 0
    else:
        viewCount = int(xmlObj.attrib['ViewCount'])

    if 'CommentCount' not in xmlObj.attrib:
        commentCount = 0
    else:
        commentCount = int(xmlObj.attrib['CommentCount'])

    if 'FavoriteCount' not in xmlObj.attrib:
        favoriteCount = 0
    else:
        favoriteCount = int(xmlObj.attrib['FavoriteCount'])

    activity = viewCount + commentCount + favoriteCount

    tags = xmlObj.attrib['Tags']
    tags = re.split('>', tags)
    for tag in tags:
        tag = tag.replace('<', '')
        if tag in langDict:
            return array([langDict[tag] * LANG_DISTANCE, activity])


def getKmeans_supportivity_vector(xmlStr):
    xmlObj = ET.fromstring(xmlStr)
    if 'AnswerCount' not in xmlObj.attrib:
        answerCount = 0
    else:
        answerCount = int(xmlObj.attrib['AnswerCount'])

    supportivity = answerCount

    tags = xmlObj.attrib['Tags']
    tags = re.split('>', tags)
    for tag in tags:
        tag = tag.replace('<', '')
        if tag in langDict:
            return array([langDict[tag] * LANG_DISTANCE, supportivity])




records = sc.textFile("/data/stackOverflow2017")
# records = sc.parallelize(['<row Id="138361019" PostId="26687223" VoteTypeId="2" CreationDate="2017-05-09T00:00:00.000" />', '<row Id="35391639" PostId="23145783" Score="0" Text="Sorry, I am still learning. I began learning 3 days ago... Still lots to learn. Thanks for the tip" CreationDate="2014-04-18T01:02:41.973" UserId="3546942" />', '<row Id="124925153" PostId="37706879" VoteTypeId="2" CreationDate="2016-10-25T00:00:00.000" />','<row Id="14351745" PostTypeId="1" AcceptedAnswerId="14354358" CreationDate="2013-01-16T05:08:36.767" Score="1" ViewCount="163" Body="&lt;p&gt;I\'m trying to figure out how to write a MySQL query that will return the nearest data which Actor = 210 for in terms E_id = 3.&lt;/p&gt;&#xA;&#xA;&lt;p&gt;This is my original table:&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;Session              Player  E_id  Time                     Actor  PosX  PosY  index&#xA;-------------------  ------  ----  -----------------------  -----  ----  ----  -----&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    906   466   6&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    342   540   7&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  31     812   244   8&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   614   9&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  21     342   688   10&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  49     812   170   11&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  50     248   466   12&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    718   318   13&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  52     154   466   14&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  211    499   250   15&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   16&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   466   17&#xA;23131885ccc560bb6c8  10125   15    01-11-2012 08:56:38.323  20     718   318   18&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  52     154   466   19&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  11     499   250   20&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   21&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;If I fire query&lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;select * from table where E_id = 3 or Actor = 210;&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;I get this result &lt;/p&gt;&#xA;&#xA;&lt;pre&gt;&lt;code&gt;Session              Player  E_id  Time                     Actor  PosX  PosY  index&#xA;-------------------  ------  ----  -----------------------  -----  ----  ----  -----&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    906   466   6&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    342   540   7&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   614   9&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    718   318   13&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   16&#xA;23131885ccc560bb6c8  10125   17    01-11-2012 08:56:38.323  210    248   466   17&#xA;23131885ccc560bb6c8  10125   3     01-11-2012 08:56:40.63   208    510   414   21&#xA;&lt;/code&gt;&lt;/pre&gt;&#xA;&#xA;&lt;p&gt;&lt;strong&gt;expected Result is:&lt;/strong&gt;&#xA; row with index no 13 for row index no 16 and &#xA; row with index no 17 for row index no 21 &lt;/p&gt;&#xA;&#xA;&lt;p&gt;Index 16 and 21 both &lt;strong&gt;e_id&lt;/strong&gt; is &lt;strong&gt;3&lt;/strong&gt;&lt;/p&gt;&#xA;" OwnerUserId="1568236" LastEditorUserId="297408" LastEditDate="2013-01-16T09:40:25.020" LastActivityDate="2013-01-17T07:15:49.120" Title="SQL Query to show nearest data to specific event" Tags="&lt;php&gt;&lt;mysql&gt;&lt;sql&gt;" AnswerCount="4" CommentCount="11" />',
#  '<row Id="85641904" PostId="6518136" VoteTypeId="2" CreationDate="2015-03-21T00:00:00.000" />'])

# FILTERING STAGE
posts = records.filter(lambda data: is_valid_post_record(data) and hasLangTags(data))
questions = records.filter(lambda data: is_valid_question_record(data) and hasLangTags(data))

# MAPPING STAGE
activity_vector = posts.map(getKmeans_activity_vector)
support_vector = questions.map(getKmeans_supportivity_vector)

# K-MEANS STAGE
activity_clusters = KMeans.train(activity_vector, CLUSTERS_NUMBER, maxIterations=10, initializationMode="random")
support_clusters = KMeans.train(support_vector, CLUSTERS_NUMBER, maxIterations=10, initializationMode="random")

# PLOTTING STAGE
#def get_plot_data ( input_data , kmeans_model):
#    result = kmeans_model.predict(input_data)
#    return [input_data, result]

#plot_activity_data = activity_vector.map(lambda data : get_plot_data(data, activity_clusters))
#plot_support_data = support_vector.map(lambda data : get_plot_data(data, support_clusters))

#fig, ax = plt.subplots()
#colourList = list(mtl.colors.cnames.iteritems())

# Evaluate clustering by computing Within Set Sum of Squared Errors
# def error(point):
#     center = clusters.centers[clusters.predict(point)]
#     return sqrt(sum([x**2 for x in (point - center)]))

# WSSSE = scoredLang.map(lambda point: error(point)).reduce(lambda x, y: x + y)
# print("Within Set Sum of Squared Error = " + str(WSSSE))


# clusters.save(sc, "KMeansModel")