import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


public class LoadTrainData {
	public HashMap<String, Double> featureDict;
	public HashMap<String, String> labelDict;
	public int size;
	public int featureNum;
	
	public LoadTrainData() {
		featureDict = new HashMap<String, Double>();
		labelDict = new HashMap<String, String>();
		size = 0;
		featureNum = 0;
	}
	public void getData(Configuration conf) throws IOException {
		String line;
		String index, label;
		String feature;
		double weight;
		featureNum = Integer.parseInt(conf.get("featureNum"));
		String filePath = conf.get("trainDataPath");
		Path path = new Path(filePath);
		FileSystem hdfs = path.getFileSystem(conf);
		FSDataInputStream fin = hdfs.open(path);
		InputStreamReader inReader = new InputStreamReader(fin);
		BufferedReader bfReader = new BufferedReader(inReader);
		while((line = bfReader.readLine()) != null) {
			String[] wholeData = line.split(", ");
			index = wholeData[0];
			label = wholeData[1];
			labelDict.put(index, label);
			for(int i = 2; i < wholeData.length; i++) {
				feature = wholeData[i].split(":")[0];
				weight = Double.parseDouble(wholeData[i].split(":")[1]);
				String temp = index + "#" + feature;
				featureDict.put(temp, weight);
			}
			size++;
		}
		bfReader.close();
		inReader.close();
		fin.close();
	}
	public ArrayList<Double> getWeights(int index) {
		//Format of featureDict: {index#feature: weight}
		ArrayList<Double> weights = new ArrayList<Double>(featureNum);
		String temp;
		for(int i = 0; i < featureNum; i++) {
			temp = String.valueOf(index) + "#" + String.valueOf(i);
			Double weight = featureDict.get(temp);
			if(weight == null) {
				weights.add(0.0);
			}
			else {
				weights.add(weight);
			}
		}
		return weights;
	}
}
