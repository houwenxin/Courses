import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;


public class NaiveBayesTest {
	public static class TestMapper extends Mapper<Object, Text, Text, Text> {
		public LoadModel model;
		public LoadInfo info;
		public void setup(Context context) throws IOException {
			Configuration conf = context.getConfiguration();
			conf.set("model", "model/part-r-00000");
			model = new LoadModel();
			model.getData(conf);
			info = new LoadInfo();
			conf.set("info", "info.txt");
			info.getInfo(conf);
		}
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Text outputKey, outputValue;	//key: id, value: class
			String[] vals;
			String temp;
			double count;
			double Pxjyi, PXyi, normFactor, Pyi;
			vals = value.toString().split(", ");
			double maxP = -100;
			int idx = -2;
			for(int i = -1; i < 2; i++) {	// For every class: negative(-1), neutral(0), positive(1)
				PXyi = 1; // Initial Probability
				normFactor = model.model_dict.get(String.valueOf(i));
				Pyi = info.Pys.get(String.valueOf(i));
				for(int j = 1; j < vals.length; j++) {
					String feature = vals[j].split(":")[0];
					count = Double.parseDouble(vals[j].split(":")[1]);
					temp = String.valueOf(i) + "#" + feature;
					//System.out.println(temp);
					Double weight = model.model_dict.get(temp);
					// To avoid nullpointer exception: P(xj | yi) does not exist.
					if(weight == null) {
						Pxjyi = 0;
					}
					else {
						Pxjyi = count * weight.doubleValue() / normFactor;
					}
					PXyi = PXyi * Pxjyi;
				}
				if(PXyi * Pyi > maxP) {
					maxP = PXyi * Pyi;
					idx = i;
				}
			}
			outputKey = new Text(vals[0]);
			outputValue = new Text(String.valueOf(idx));
			context.write(outputKey, outputValue);
		}
	}
}
