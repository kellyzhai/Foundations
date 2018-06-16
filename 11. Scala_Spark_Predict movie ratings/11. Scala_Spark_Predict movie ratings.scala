// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC Please submit your answers by publishing a databricks notebook here:
// MAGIC https://docs.google.com/forms/d/e/1FAIpQLSeUQ4D7W_APiZHnQhSHmaZ0WPykmdflivEH0rt6ZlxJncWGzg/viewform?usp=sf_link

// COMMAND ----------

// MAGIC %md # Movies Rating's model:
// MAGIC 
// MAGIC Input dataset:
// MAGIC 
// MAGIC ```/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv```
// MAGIC 
// MAGIC Using the Apache Spark ML pipeline, <b>build a model to predict the rating of a movie </b> based on the available features. How would you handle non-numerical data?
// MAGIC 
// MAGIC Information about the dataset:
// MAGIC  - https://cran.r-project.org/web/packages/ggplot2movies/ggplot2movies.pdf
// MAGIC  - You can find plenty of exploratory analysis examples around the web for this particular dataset
// MAGIC 
// MAGIC Read https://spark.apache.org/docs/latest/ml-features.html to learn more about transforming features, dealing with categorical variables, etc.

// COMMAND ----------



import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions
import org.apache.spark.sql.types.DoubleType
//val sqlContext = new org.apache.spark.sql.SQLContext(sc);import sqlContext.implicits._
import sys.process._

val dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/movies.csv"
val movies = sqlContext.read.format("com.databricks.spark.csv")
  .option("header","true")
  .option("inferSchema", "true")
  .load(dataPath)

// COMMAND ----------

display(movies)

// COMMAND ----------

// MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/csv/ggplot2/

// COMMAND ----------

// Register spark SQL tables
movies.createOrReplaceTempView("movies")

// COMMAND ----------

// MAGIC %sql 
// MAGIC DROP TABLE IF EXISTS cleaned_movies;
// MAGIC 
// MAGIC CREATE TABLE cleaned_movies AS
// MAGIC SELECT title,rating,votes,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,Action,Animation,Comedy,Drama,Documentary,Romance,Short 
// MAGIC FROM movies

// COMMAND ----------

sqlContext.cacheTable("cleaned_movies")

// Convert back to a dataset from a table
val cleanedMovies = spark.sql("SELECT * FROM cleaned_movies")

display(cleanedMovies)



// COMMAND ----------


val summedMovies = cleanedMovies
  .groupBy("rating")
  .count() // numbers of rating per movie
  .sort("rating") 
println(cleanedMovies.count())


// COMMAND ----------

display(summedMovies)

// COMMAND ----------

val prepped = cleanedMovies.na.fill(0)
display(prepped)


// COMMAND ----------

val nonFeatureCols = Array("title", "rating","Action","Animation","Comedy","Drama","Documentary","Romance","Short")
val featureCols = prepped.columns.diff(nonFeatureCols)

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val finalPrep = assembler.transform(prepped)
display(finalPrep)

// COMMAND ----------

val Array(training, test) = finalPrep.randomSplit(Array(0.7, 0.3))

// Going to cache the data to make sure things stay snappy!
training.cache()
test.cache()

println(training.count())  
println(test.count())

// COMMAND ----------


import org.apache.spark.ml.regression.LinearRegression

val lrModel = new LinearRegression()
  .setLabelCol("rating")
  .setFeaturesCol("features")
  .setElasticNetParam(0.5)

println("Printing out the model Parameters:")
println("-"*20)
println(lrModel.explainParams)
println("-"*20)

// COMMAND ----------

import org.apache.spark.mllib.evaluation.RegressionMetrics
val lrFitted = lrModel.fit(training)

// COMMAND ----------

val holdout = lrFitted
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction,1)) as prediction", 
    "rating", 
    """CASE double(round(prediction)) = rating 
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout)

// COMMAND ----------

display(holdout.selectExpr("sum(equal)/sum(1)"))

// COMMAND ----------

// have to do a type conversion for RegressionMetrics
val rm = new RegressionMetrics(
  holdout.select("prediction", "rating").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm.meanSquaredError)
println("MAE: " + rm.meanAbsoluteError)
println("RMSE Squared: " + rm.rootMeanSquaredError)
println("R Squared: " + rm.r2)
println("Explained Variance: " + rm.explainedVariance + "\n")

// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.ml.evaluation.RegressionEvaluator

import org.apache.spark.ml.{Pipeline, PipelineStage}

val rfModel = new RandomForestRegressor()
  .setLabelCol("rating")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder()
  .addGrid(rfModel.maxDepth, Array(5, 10))
  .addGrid(rfModel.numTrees, Array(20, 60))
  .build()


val steps:Array[PipelineStage] = Array(rfModel)

val pipeline = new Pipeline().setStages(steps)

val cv = new CrossValidator() // 
  .setEstimator(pipeline) // 
  .setEstimatorParamMaps(paramGrid)
  .setEvaluator(new RegressionEvaluator().setLabelCol("rating"))

val pipelineFitted = cv.fit(training)

// COMMAND ----------

println("The Best Parameters:\n--------------------")
println(pipelineFitted.bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(0))
pipelineFitted
  .bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
  .stages(0)
  .extractParamMap

// COMMAND ----------

val holdout2 = pipelineFitted.bestModel
  .transform(test)
  .selectExpr("prediction as raw_prediction", 
    "double(round(prediction,1)) as prediction", 
    "rating", 
    """CASE double(round(prediction)) = rating
  WHEN true then 1
  ELSE 0
END as equal""")
display(holdout2)

// COMMAND ----------

// have to do a type conversion for RegressionMetrics
val rm2 = new RegressionMetrics(
  holdout2.select("prediction", "rating").rdd.map(x =>
  (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

println("MSE: " + rm2.meanSquaredError)
println("MAE: " + rm2.meanAbsoluteError)
println("RMSE Squared: " + rm2.rootMeanSquaredError)
println("R Squared: " + rm2.r2)
println("Explained Variance: " + rm2.explainedVariance + "\n")

// COMMAND ----------

display(holdout2.selectExpr("sum(equal)/sum(1)"))

// COMMAND ----------


