import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;


public class KNN {
	public static class KNNMapper extends Mapper<Object, Text, Text, Text> {
		public LoadTrainData trainData;
		public ArrayList<ArrayList<Double>> trainDataWeights;
		public void setup(Context context) throws IOException {
			Configuration conf = context.getConfiguration();
			conf.set("trainDataPath", "traindata/traindata.txt");
			trainData = new LoadTrainData();
			trainData.getData(conf);
			trainDataWeights = new ArrayList<ArrayList<Double>>(trainData.size);
			for(int i = 0; i < trainData.size; i++){
				trainDataWeights.add(i, trainData.getWeights(i));
			}
		}
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			int k = Integer.parseInt(conf.get("k"));
			
			String[] featurePair;
			String feature;
			double weight;
			ArrayList<Double> testDataWeights = new ArrayList<Double>(trainData.featureNum);
			HashMap<String, Double> dataDict = new HashMap<String, Double>();
			String[] line = value.toString().split(", ");
			String index = line[0];
			for(int i = 1; i < line.length; i++) {
				featurePair = line[i].split(":");
				feature = featurePair[0];
				weight = Double.parseDouble(featurePair[1]);
				dataDict.put(feature, weight);
			}
			for(int i = 0; i < trainData.featureNum; i++) {
				Double tempWeight = dataDict.get(String.valueOf(i));
				if(tempWeight == null) {
					testDataWeights.add(0.0);
				}
				else {
					testDataWeights.add(tempWeight);
				}
			}
			ArrayList<String> KNN = getKNN(trainDataWeights, testDataWeights, k);
			for(int i = 0; i < k; i++) {
				String trainDataIndex = KNN.get(i);
				String trainDataLabel = trainData.labelDict.get(trainDataIndex);
				context.write(new Text(index), new Text(trainDataIndex + "#" + trainDataLabel));
			}
		}
		
		private ArrayList<String> getKNN(ArrayList<ArrayList<Double>> trainDataWeights, ArrayList<Double> testDataWeights, int k) {
			ArrayList<String> KNN = new ArrayList<String>(k);
			ArrayList<Double> distances = new ArrayList<Double>(k);
			double distance;
			for(int i = 0; i < k; i++) {
				KNN.add("-2");
				distances.add(Double.MAX_VALUE);
			}
			for(int i = 0; i < trainData.size; i++) {
				distance = getDistance(trainDataWeights.get(i), testDataWeights, trainData.featureNum);
				for(int j = 0; j < k; j++) {
					if(distance < distances.get(j)) {
						distances.set(j, distance);
						KNN.set(j, String.valueOf(i));
						break;
					}
				}
			}
			return KNN;
		}
		private double getDistance(ArrayList<Double> vector1, ArrayList<Double> vector2, int dimension) {
			double distance = 0.0;
			for(int i = 0; i < dimension; i++) {
				distance = distance + Math.pow((vector1.get(i) - vector2.get(i)), 2);
			}
			distance = Math.sqrt(distance);
			return distance;
		}
	}
	public static class KNNReducer extends Reducer<Text, Text, Text, Text> {
		@Override //如果Reducer没用上可以在这里加上@Override检查reduce是否有错误
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			String indexOfTrainData, label;
			HashMap<String, Integer> counter = new HashMap<String, Integer>();
			for(Text value : values) {
				if(counter.containsKey(value.toString())) {
					counter.put(value.toString(), counter.get(value.toString()) + 1);
				}
				else {
					counter.put(value.toString(), new Integer(1));
				}
			}
			// 找到出现次数最多的train data对应的index
			List<Map.Entry<String,Integer>> list = new ArrayList(counter.entrySet());
		    Collections.sort(list, (o1, o2) -> (o1.getValue() - o2.getValue()));
		    indexOfTrainData = list.get(list.size() - 1).getKey();
		    label = indexOfTrainData.split("#")[1];
		    context.write(new Text(key), new Text(label));
		}
	}
}
