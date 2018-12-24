from pyspark.ml.linalg import Vectors
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.sql.session import SparkSession
from pyspark.sql import Row

import time # To calculate run time

def f(x):
	rel = {}
	rel['features'] = Vectors.dense(float(x[0]), float(x[1]))
	return rel

if __name__ == "__main__":
	sc = SparkContext(appName="KMeans")
	spark = SparkSession(sc)
	
	start_time = time.clock()

	data = sc.textFile("Instance.txt").map(lambda line: line.split(',')).map(lambda p: Row(**f(p)))
	df = data.toDF()
	# Max iteration = 3 for comparison with Hadoop.
	kmeans = KMeans().setK(5).setMaxIter(3).setFeaturesCol('features').setPredictionCol('prediction')
	#kmeans = KMeans().setK(5).setFeaturesCol('features').setPredictionCol('prediction')
	model = kmeans.fit(df)
	centers = model.clusterCenters()
	print(centers)
	results = model.transform(df).collect()
	end_time = time.clock()
	for item in results:
     		print(str(item[0])+ ',' +  str(item[1]))
	print("Run time For KMeans on Spark: " + str(end_time - start_time) + "seconds")

