import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;


public class LoadModel {
	public HashMap<String, Double> model_dict;
	public LoadModel() {
		model_dict = new HashMap<String, Double>();
	}
	public void getData(Configuration conf) throws IOException {
		String line;
		String modelPath = conf.get("model");
		Path path = new Path(modelPath);
		FileSystem hdfs = path.getFileSystem(conf);
		FSDataInputStream fin = hdfs.open(path);
		InputStreamReader inReader = new InputStreamReader(fin);
		BufferedReader bfReader = new BufferedReader(inReader);
		while((line = bfReader.readLine()) != null) {
			String[] res = line.split("\\t");
			model_dict.put(res[0], new Double(res[1]));
			System.out.println(res[0] + ":" + new Double(res[1]));
		}
		bfReader.close();
		inReader.close();
		fin.close();
	}
}
