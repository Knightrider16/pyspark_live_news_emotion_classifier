import joblib
from pyspark.ml.pipeline import PipelineModel
import numpy as np
from sklearn.linear_model import LogisticRegression

# ------------------ Load Spark model ------------------
spark_model = PipelineModel.load("models/spark_linear_svc_pipeline")

# ------------------ Extract stages ------------------
tokenizer = spark_model.stages[0]
stopwords_remover = spark_model.stages[1]
hashingTF = spark_model.stages[2]
idf_model = spark_model.stages[3]
svc_model = spark_model.stages[4]  # LinearSVC (binary)

# ------------------ Save StopWords ------------------
stopwords_list = stopwords_remover.getStopWords()

# ------------------ Extract weights from LinearSVC ------------------
coefs = np.array(svc_model.coefficients).reshape(1, -1)
intercept = np.array([svc_model.intercept])

# ------------------ Create dummy LogisticRegression ------------------
dummy_X = np.zeros((2, hashingTF.getNumFeatures()))
dummy_y = np.array([0, 1])
sk_model = LogisticRegression()
sk_model.fit(dummy_X, dummy_y)

# ------------------ Assign coefficients ------------------
sk_model.coef_ = coefs
sk_model.intercept_ = intercept
sk_model.classes_ = np.array([0, 1])

# ------------------ Save everything for Python inference ------------------
joblib.dump({
    "model": sk_model,
    "num_features": hashingTF.numFeatures,
    "stopwords": stopwords_list
}, "models/sk_model_full.pkl")

print("✅ Spark binary model converted to scikit-learn with preprocessing!")
