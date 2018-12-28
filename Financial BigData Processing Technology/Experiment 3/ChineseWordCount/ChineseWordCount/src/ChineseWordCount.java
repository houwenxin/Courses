import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

public class ChineseWordCount {

  public static class TokenizerMapper extends
      Mapper<Object, Text, Text, IntWritable> {

    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();
    
    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
 
      HanLP.Config.ShowTermNature = false;
      
      // Used for reading download_data
      //String[] tmp = value.toString().trim().split("\\s{2}");
      //String str = tmp[3].replace(" ", "");
      // Used for reading fulldata.txt
      String[] tmp = value.toString().trim().split("\\t");
      String str = tmp[4].replace(" ", "");
      
      List<Term> list = StandardTokenizer.segment(str);
      CoreStopWordDictionary.apply(list);
    
      for(int i = 0; i < list.size(); i++){
    	word.set(list.get(i).toString());
    	context.write(word, one);
      }
    }
  }

  public static class IntSumReducer extends
      Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      int k = context.getConfiguration().getInt("k", 0);
      for (IntWritable val : values) {
        sum += val.get();
      }
      if(sum > k){
    	result.set(sum);
        context.write(key, result);
      }
    }
  }
  private static class IntWritableDecreasingComparator extends IntWritable.Comparator{
	  public int compare(WritableComparable a, WritableComparable b){
		  return -super.compare(a, b);
	  }
	  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2){
		  return -super.compare(b1, s1, l1, b2, s2, l2);
	  }
  }
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs =
        new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 3) {
      System.err.println("Usage: ChineseWordCount <in> <out> <k>");
      System.exit(2);
    }
    Path tempDir = new Path("ChineseWordCount-temp-" + Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
    conf.setInt("k", Integer.parseInt(otherArgs[2]));
    
    Job job = new Job(conf, "word count");
    job.setJarByClass(ChineseWordCount.class);
    try{
    job.setMapperClass(TokenizerMapper.class);
    job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(IntSumReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job, tempDir);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    //System.exit(job.waitForCompletion(true) ? 0 : 1);
    if(job.waitForCompletion(true)){
    	Job sortJob = new Job(conf, "sort");
    	sortJob.setJarByClass(ChineseWordCount.class);
    	FileInputFormat.addInputPath(sortJob, tempDir);
    	sortJob.setInputFormatClass(SequenceFileInputFormat.class);
    	sortJob.setMapperClass(InverseMapper.class);
    	sortJob.setNumReduceTasks(1);
    	FileOutputFormat.setOutputPath(sortJob, new Path(otherArgs[1]));
    	sortJob.setOutputKeyClass(IntWritable.class);
    	sortJob.setOutputValueClass(Text.class);
    	sortJob.setSortComparatorClass(IntWritableDecreasingComparator.class);
    	System.exit(sortJob.waitForCompletion(true) ? 0:1);
    }
    }
    finally{
    	FileSystem.get(conf).deleteOnExit(tempDir);
    }
  }
}
