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
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.filecache.DistributedCache;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.InverseMapper;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

public class InvertedIndex {

  public static class TokenizerMapper extends
      Mapper<Object, Text, Text, Text> {

    //private final static IntWritable one = new IntWritable(1);
    private Text outputKey = new Text();
    private Text outputValue = new Text();
    
    public void map(Object key, Text value, Context context)
        throws IOException, InterruptedException {
      
      HanLP.Config.ShowTermNature = false;
      
      // Used for reading download_data
      //String[] tmp = value.toString().trim().split("\\s{2}");
      //String str = tmp[3].replace(" ", "");
      // Used for reading fulldata.txt
      String[] tmp = value.toString().trim().split("\\t");
      String strCode = tmp[0];
      String strURL = tmp[5];
      String strTitle = tmp[4].replace(" ", "");
      
      List<Term> listTitleWords = StandardTokenizer.segment(strTitle);
      CoreStopWordDictionary.add("%");
      CoreStopWordDictionary.add("$");
      CoreStopWordDictionary.add("-");
      CoreStopWordDictionary.add("%-");
      CoreStopWordDictionary.apply(listTitleWords);
      for(int i = 0; i < listTitleWords.size(); i++){
    	outputKey.set(listTitleWords.get(i).toString() + "#" + strCode);
    	outputValue.set(strURL + "#" + "1");
    	context.write(outputKey, outputValue);
      }
    }
  }

  public static class MainReducer extends
      Reducer<Text, Text, Text, IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<Text> values, Context context)
        throws IOException, InterruptedException {
      int sum = 0;
      Text word = new Text();
      String titleWord = key.toString().split("#")[0];
      String code = key.toString().split("#")[1];
      String URL;
      StringBuilder strBuilder = new StringBuilder("[");
      int Occurrence;
      for (Text val : values) {
    	URL = val.toString().split("#")[0];
    	strBuilder.append(URL+", ");
    	Occurrence = Integer.parseInt(val.toString().split("#")[1]);
        sum += Occurrence;
      }
      strBuilder.replace(strBuilder.length()-2, strBuilder.length(), "]");
      word.set(titleWord + "#" + code + "#" + strBuilder.toString());
      result.set(sum);
      context.write(word, result);
    }
  }

  private static class SecondSortComparator extends WritableComparator{
	  protected SecondSortComparator(){
	  	  super(Text.class, true);
	  }
	  @Override
	  public int compare(WritableComparable a, WritableComparable b) {
			Text o1 = (Text) a;
			Text o2 = (Text) b;
			int occurrence1 = Integer.parseInt(o1.toString().split("#")[1]);
			String titleWord1 = o1.toString().split("#")[0];
			int occurrence2 = Integer.parseInt(o2.toString().split("#")[1]);
			String titleWord2 = o2.toString().split("#")[0];
			if(! titleWord1.equals(titleWord2))
				return titleWord1.compareTo(titleWord2);
			else
				return occurrence2 - occurrence1;
		}
  }
  public static class SecondSortMapper extends Mapper<Text, IntWritable, Text, NullWritable>{
	  private String titleWord, occurrence, code, URL;
	  public void map(Text key, IntWritable value, Context context) throws IOException, InterruptedException{
		  //Input Format: titleWord#code#[url0, url1, ...]		Occurrences
		  //Output Format: titleWord#Occurrence#code#[URL]		Null
		  titleWord = key.toString().split("#")[0];
		  occurrence = value.toString();
		  code = key.toString().split("#")[1];
		  URL = key.toString().split("#")[2];
		  context.write(new Text(titleWord+"#"+occurrence+"#"+code+"#"+URL), NullWritable.get());
	  }
  }
  public static class SecondSortPartitioner extends HashPartitioner<Text, NullWritable>
  {
	  //Partitioned by titleWord
	  public int getPartition(Text key, NullWritable value, int numReduceTasks){
		  String term = new String();
		  term = key.toString().split("#")[0];
		  return super.getPartition(new Text(term), value, numReduceTasks);
	  }
  }
  public static class SecondSortReducer extends Reducer<Text, NullWritable, Text, NullWritable>{
	  private Text word = new Text();
	  @Override
	  public void reduce(Text key, Iterable<NullWritable> value, Context context) throws IOException, InterruptedException{
		  word.set(key.toString().replaceAll("#", ", "));
		  context.write(word, NullWritable.get());
	  }
  }
  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String[] otherArgs =
        new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
      System.err.println("Usage: InvertedIndex <in> <out>");
      System.exit(2);
    }
    Path tempDir = new Path("InvertedIndex-temp-" + Integer.toString(new Random().nextInt(Integer.MAX_VALUE)));
    //conf.setInt("k", Integer.parseInt(otherArgs[2]));
    
    Job job = new Job(conf, "inverted index");
    job.setJarByClass(InvertedIndex.class);
    try{
    job.setMapperClass(TokenizerMapper.class);
    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(Text.class);
    //job.setCombinerClass(IntSumReducer.class);
    job.setReducerClass(MainReducer.class);
    //job.setPartitionerClass(NewPartitioner.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job, tempDir);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    //System.exit(job.waitForCompletion(true) ? 0 : 1);
    if(job.waitForCompletion(true)){
    	Job sortJob = new Job(conf, "sort");
    	sortJob.setJarByClass(InvertedIndex.class);
    	FileInputFormat.addInputPath(sortJob, tempDir);
    	sortJob.setInputFormatClass(SequenceFileInputFormat.class);
    	sortJob.setMapperClass(SecondSortMapper.class);
    	sortJob.setMapOutputKeyClass(Text.class);
        sortJob.setMapOutputValueClass(NullWritable.class);
    	sortJob.setPartitionerClass(SecondSortPartitioner.class);
    	
    	sortJob.setReducerClass(SecondSortReducer.class);
    	sortJob.setNumReduceTasks(1);
    	FileOutputFormat.setOutputPath(sortJob, new Path(otherArgs[1]));
    	sortJob.setOutputKeyClass(Text.class);
    	sortJob.setOutputValueClass(NullWritable.class);
    	sortJob.setSortComparatorClass(SecondSortComparator.class);;
    	//sortJob.setGroupingComparatorClass(SecondSortComparator.class);
    	System.exit(sortJob.waitForCompletion(true) ? 0:1);
    }
    }
    finally{
    	FileSystem.get(conf).deleteOnExit(tempDir);
    }
  }
}