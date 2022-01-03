import numpy as np
from joblib import dump, load
import os.path
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import *
from pyspark.ml.feature import Tokenizer,  StopWordsRemover, RegexTokenizer, CountVectorizer, HashingTF, IDF
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans

sc = SparkContext("local[2]", "Spam")
ssc = StreamingContext(sc, 1)
spark = SparkSession.builder.appName("spamClassifier").getOrCreate()

lines = ssc.socketTextStream("localhost", 7000)
mySchema = StructType([ StructField("id", StringType(), True),StructField("message", StringType(), True),StructField("spam/ham", StringType(), True)])


mnb = MultinomialNB()


def multinomialNB(X,y,mnb):
    if (os.path.isfile('multinomialNB.joblib')):
        mnb = load('multinomialNB.joblib')  
    mnb.partial_fit(X, y,classes=[1,0])
    dump(mnb, 'multinomialNB.joblib')
    


def preProcess(df):
    df = df.filter("subject is not null").filter("body is not null").filter("`spam/ham` is not null")
    messageTokenizer = Tokenizer(inputCol="message", outputCol="messageTokens")
    df = messageTokenizer.transform(df)
    remover = StopWordsRemover(inputCol="messageTokens", outputCol="filtered")
    df =remover.transform(df)
    final=[]
    for row in df.rdd.toLocalIterator():
        final.append({"id":row['id'],"processed":[word for word in row['filtered'] if word.isalpha()]})
    df1 = spark.createDataFrame(final)
    df = df.join(df1, df.id == df1.id).drop(df1.id)
    countVectorizer = CountVectorizer (inputCol="processed", outputCol="counts", vocabSize=3000)
    model = countVectorizer.fit(df)
    df = model.transform(df)
    return df

def convert(df):
    X_train = np.array(df.select("counts").collect(),dtype='object')
    Y_train = np.array(df.select("spam/ham").collect(),dtype='int')
    X_train = np.squeeze(X_train)
    Y_train = np.squeeze(Y_train)
    return X_train,Y_train
    
def batch(rdd):
    df=spark.read.json(rdd)
    row= df.head()
    if row:
        final=[]
        for i in row.__fields__:
            if (row[i]['feature2']=='spam'):
                value = 1
            else:
                value = 0
            final.append({"id":i,"message":str(row[i]['feature0'])+' '+str(row[i]['feature1']),"spam/ham":value})
        df = spark.createDataFrame(final,schema=mySchema)
        df = preProcess(df)
        X_train,Y_train=convert(df)
        multinomialNB(X_train,Y_train,mnb)
    
lines.foreachRDD(lambda rdd: batch(rdd))

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
