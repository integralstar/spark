from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
      .appName("Heart disease Classifier") \
      .config("spark.executor.memory", "70g") \
      .config("spark.driver.memory", "50g") \
      .config("spark.memory.offHeap.enabled", True) \
      .config("spark.memory.offHeap.size", "16g") \
      .getOrCreate()

heart_df = spark.read.format("csv").options(header="true", inferschema="true", sep=',').load("heart.csv")

vectors = VectorAssembler(inputCols = ['age','sex', 'cp','trestbps','chol','fbs', 'restecg', 'thalach', 'exang','oldpeak', 'slope', 'ca', 'thal'], outputCol = 'features')

vheart_df = vectors.transform(heart_df)

vheart_df = vheart_df.select(['features', 'target'])

vheart_df.show(5)

train_df, test_df = vheart_df.randomSplit([0.75, 0.25])

classifier= RandomForestClassifier(featuresCol = 'features', labelCol='target')

model = classifier.fit(train_df)

modelSummary = model.summary

model_predictions = model.transform(test_df)
model_predictions.select("features", "target", "prediction").show(7)

accuracy = modelSummary.accuracy
fPR = modelSummary.weightedFalsePositiveRate
tPR = modelSummary.weightedTruePositiveRate
fMeasure = modelSummary.weightedFMeasure()
precision = modelSummary.weightedPrecision
recall = modelSummary.weightedRecall

print("Accuracy: ", accuracy)
print("False Positive Rate : ", fPR)
print("True Positive Rate : ", tPR)
print("F : ", fMeasure)
print("Precision : ", precision)
print("Recall : ", recall)

spark.stop()