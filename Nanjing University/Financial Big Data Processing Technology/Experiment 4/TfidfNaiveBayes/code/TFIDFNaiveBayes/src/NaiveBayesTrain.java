import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;


public class NaiveBayesTrain {
	
	public static class TrainMapper extends Mapper<Object, Text, Text, DoubleWritable> {
		private Text outputKey = new Text();
		private DoubleWritable outputValue = new DoubleWritable();
		
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] tmp = value.toString().split(", ");
			String label;
			String[] featurePair;
			String feature;
			double weight;
			label = tmp[1];
			for(int i = 2; i < tmp.length; i++){
				featurePair = tmp[i].split(":");
				feature = featurePair[0];
				weight = Double.parseDouble(featurePair[1]);
				outputKey.set(label + "#" + feature);
				outputValue.set(weight);
				context.write(outputKey, outputValue);
				// To calculate P(Y)
				outputKey.set(label);
				context.write(outputKey, outputValue);
			}
		}
	}
	public static class TrainReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
		DoubleWritable outputValue = new DoubleWritable();
		public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
			double sum = 0;
			for(DoubleWritable value : values){
				sum += value.get();
			}
			outputValue.set(sum);
			context.write(key, outputValue);
		}
	}
	
}

