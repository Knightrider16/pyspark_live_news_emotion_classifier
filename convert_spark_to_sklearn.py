# convert_spark_to_sklearn_full.py
import joblib
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load Spark model
spark_model = PipelineModel.load("models/spark_linear_svc_pipeline")

# Extract stages
tokenizer = spark_model.stages[1]       # Tokenizer
stopwords_remover = spark_model.stages[2]  # StopWordsRemover
hashingTF = spark_model.stages[3]       # HashingTF
idf_model = spark_model.stages[4]       # IDF
ovr_model = spark_model.stages[5]       # OneVsRest
svc_models = ovr_model.models           # List of LinearSVCModel for each class

# Save StopWords
stopwords_list = stopwords_remover.getStopWords()

# Extract weights for multi-class LinearSVC
coefs = []
intercepts = []
for svc in svc_models:
    coefs.append(np.array(svc.coefficients))
    intercepts.append(svc.intercept)

# Create sklearn logistic regression model
dummy_X = np.zeros((1, hashingTF.getNumFeatures()))
dummy_y = np.zeros(1)
sk_model = LogisticRegression()
sk_model.fit(dummy_X, dummy_y)  # Dummy fit

# Assign coefficients
sk_model.coef_ = np.vstack(coefs)
sk_model.intercept_ = np.array(intercepts)
sk_model.classes_ = np.arange(len(coefs))

# Save everything needed for Python inference
joblib.dump({
    "model": sk_model,
    "num_features": hashingTF.getNumFeatures(),
    "stopwords": stopwords_list
}, "models/sk_model_full.pkl")

print("✅ Spark model converted to scikit-learn with preprocessing!")
