# train_model_spark.py
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LinearSVC, OneVsRest
import os

# ------------------ Windows-specific fixes ------------------
# Python path
os.environ["PYSPARK_PYTHON"] = "C:/Users/Gautham/AppData/Local/Programs/Python/Python313/python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Users/Gautham/AppData/Local/Programs/Python/Python313/python.exe"

# Hadoop path
os.environ["HADOOP_HOME"] = "C:/hadoop"
os.environ["PATH"] += os.pathsep + "C:/hadoop/bin"

# ------------------ Spark session ------------------
spark = SparkSession.builder.appName("NewsEmotionTrain").getOrCreate()

# ------------------ Load dataset ------------------
df = spark.read.csv("data/emotions.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("text", "headline")

# ------------------ Label indexing ------------------
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

# ------------------ Text preprocessing ------------------
tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
stopwords = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

# ------------------ Multi-class LinearSVC using OneVsRest ------------------
svc = LinearSVC(featuresCol="features", labelCol="label_index", maxIter=20, regParam=0.1)
ovr = OneVsRest(classifier=svc, labelCol="label_index", featuresCol="features")

# ------------------ Pipeline ------------------
pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords, hashingTF, idf, ovr])

# ------------------ Train/test split ------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# ------------------ Train ------------------
model = pipeline.fit(train_df)

# ------------------ Save pipeline ------------------
model.write().overwrite().save("spark_linear_svc_pipeline")

print("Pipeline saved successfully!")
