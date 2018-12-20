import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class NaiveBayesMain {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		Configuration conf = new Configuration();
	    String[] otherArgs =
	        new GenericOptionsParser(conf, args).getRemainingArgs();
	    if (otherArgs.length != 3) {
	      System.err.println("Usage: NaiveBayesTrain <in> <out> <train / test>");
	      System.exit(2);
	    }
	    if(otherArgs[2].equals("train")) {
	    	Job job = new Job(conf, "NaiveBayesTrain");
	    	job.setJarByClass(NaiveBayesTrain.class);
	    	job.setMapperClass(NaiveBayesTrain.TrainMapper.class);
	    	job.setMapOutputKeyClass(Text.class);
	    	job.setMapOutputValueClass(DoubleWritable.class);
	    	job.setReducerClass(NaiveBayesTrain.TrainReducer.class);
	    	job.setOutputKeyClass(Text.class);
	    	job.setOutputValueClass(DoubleWritable.class);
	    	FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    	FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    	System.exit(job.waitForCompletion(true) ? 0 : 1);
	    }
	    else if(otherArgs[2].equals("test")) {
	    	Job job = new Job(conf, "NaiveBayesTest");
	    	job.setJarByClass(NaiveBayesTest.class);
	    	job.setMapperClass(NaiveBayesTest.TestMapper.class);
	    	job.setMapOutputKeyClass(Text.class);
	    	job.setMapOutputValueClass(Text.class);
	    	FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
	    	FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
	    	System.exit(job.waitForCompletion(true) ? 0 : 1);
	    }
	}

}

