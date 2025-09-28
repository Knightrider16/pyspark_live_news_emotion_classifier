from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LinearSVC

# ------------------ Spark session ------------------
spark = SparkSession.builder.appName("NewsSentimentBinaryTrain").getOrCreate()

# ------------------ Load dataset ------------------
df = spark.read.csv("data/emotions.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("text", "headline")

# ------------------ Map multi-class to binary ------------------
# Consider labels 0,3,4,5 as Negative (0) and 1,2 as Positive (1)
from pyspark.sql.functions import when, col
df = df.withColumn("binary_label", when(col("label").isin([1,2]), 1).otherwise(0))

# ------------------ Pipeline ------------------
tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
stopwords = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
svc = LinearSVC(featuresCol="features", labelCol="binary_label", maxIter=50, regParam=0.1)

pipeline = Pipeline(stages=[tokenizer, stopwords, hashingTF, idf, svc])

# ------------------ Train ------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# ------------------ Save Spark pipeline ------------------
model.write().overwrite().save("models/spark_linear_svc_pipeline")
print("✅ Spark binary sentiment pipeline saved!")