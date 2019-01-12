import numpy as np
from pyspark import SparkContext
import time

def closestPoint(p, centers):
	index = 0
	distance = float("+inf")
	for i in range(len(centers)):
		tempDistance = np.sum((p - centers[i]) ** 2)
		if tempDistance < distance:
			distance = tempDistance
			index = i
	return index

if __name__ == "__main__":
	start = time.clock() # Start time

	sc = SparkContext(appName="KMeans")
	data = sc.textFile("Instance.txt").map(lambda line: np.array([float(x) for x in line.split(",")]))
	K = 5
	iterationNum = 3	
	centers = data.takeSample(False, K, 1)
	for repetition in range(iterationNum):
		closestPoints = data.map(lambda p: (closestPoint(p, centers), (p, 1)))
		pointStats = closestPoints.reduceByKey(lambda x1, x2: (x1[0] + x2[0], x1[1] + x2[1]))
		newPoints = pointStats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
		for index, p in newPoints:
			centers[index] = p
	newInput = closestPoints.collect()

	end = time.clock() # End time

	for p in newInput:
		print(str(p[1][0])+ "\t" +str(p[0]))
	print("Run time for KMeans on Spark: " + str(end - start) + "seconds")
