from pyspark.ml.linalg import Vectors
from pyspark import SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.sql.session import SparkSession
from pyspark.sql import Row

def f(x):
	rel = {}
	rel['features'] = Vectors.dense(float(x[0]), float(x[1]))
	return rel

if __name__ == "__main__":
	sc = SparkContext(appName="KMeans")
	spark = SparkSession(sc)

	data = sc.textFile("Instance.txt").map(lambda line: line.split(',')).map(lambda p: Row(**f(p)))
	df = data.toDF()
	kmeans = KMeans().setK(5).setFeaturesCol('features').setPredictionCol('prediction')
	model = kmeans.fit(df)
	centers = model.clusterCenters()
	print(centers)
	results = model.transform(df).collect()
	for item in results:
     		print(str(item[0])+ ',' +  str(item[1]))
	

