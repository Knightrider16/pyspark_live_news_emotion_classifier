from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LinearSVC, OneVsRest

# ------------------ Spark session ------------------
spark = SparkSession.builder.appName("NewsEmotionTrain").getOrCreate()

# ------------------ Load dataset ------------------
df = spark.read.csv("data/emotions.csv", header=True, inferSchema=True)
df = df.withColumnRenamed("text", "headline")

# ------------------ Pipeline ------------------
label_indexer = StringIndexer(inputCol="label", outputCol="label_index")
tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
stopwords = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
hashingTF = HashingTF(inputCol="filtered_tokens", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
svc = LinearSVC(featuresCol="features", labelCol="label_index", maxIter=20, regParam=0.1)
ovr = OneVsRest(classifier=svc, labelCol="label_index", featuresCol="features")

pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords, hashingTF, idf, ovr])

# ------------------ Train ------------------
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train_df)

# ------------------ Save Spark pipeline ------------------
model.write().overwrite().save("models/spark_linear_svc_pipeline")
print("Spark pipeline saved!")
