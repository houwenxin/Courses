import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class LoadInfo {
	public HashMap<String, Double> Pys;
	public LoadInfo() {
		Pys = new HashMap<String, Double>();
	}
	public void getInfo(Configuration conf) throws IOException {
		String filePath = conf.get("info");
		String line;
		String Pyi, labeli;
		Path infoPath = new Path(filePath);
		FileSystem hdfs = infoPath.getFileSystem(conf);
		FSDataInputStream fin = hdfs.open(infoPath);
		InputStreamReader inReader = new InputStreamReader(fin);
		BufferedReader bfReader = new BufferedReader(inReader);
		while((line = bfReader.readLine()) != null) {
			labeli = line.split(":")[1];
			Pyi = line.split(":")[2];
			Pys.put(labeli, new Double(Pyi));
		}
		bfReader.close();
		inReader.close();
		fin.close();
	}
}
