import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.HTableDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.client.Admin;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.HBaseAdmin;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Table;
import org.apache.hadoop.hbase.util.Bytes;


public class HBaseStudy
{
	private static Configuration conf = null;
	static
	{
		conf = HBaseConfiguration.create();
	}
	// Create a table
	public static void createTable(TableName tableName, String[] familys) throws Exception
	{

		Connection conn = ConnectionFactory.createConnection(conf);
		Admin admin = conn.getAdmin();
		HTableDescriptor t = new HTableDescriptor(tableName);
		for(int i = 0; i < familys.length; i++){
			t.addFamily(new HColumnDescriptor(familys[i]));
		}
		admin.createTable(t);
		System.out.println("Create Table "+ tableName);
	}
	// Add data
	public static void addData(TableName tableName, String rowKey, String family, String qualifier, String value)
			throws Exception
	{
		try {
			Connection conn = ConnectionFactory.createConnection(conf);
			Table table = conn.getTable(tableName);
			Put put = new Put(Bytes.toBytes(rowKey));
			
			put.addColumn(Bytes.toBytes(family), Bytes.toBytes(qualifier), Bytes.toBytes(value));
			table.put(put);
			System.out.println("Insert record successfully");
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	public static void dropTable(TableName tableName) throws IOException {
		Connection conn = ConnectionFactory.createConnection();
		Admin admin = conn.getAdmin();
		if(admin.tableExists(tableName)) {
			admin.disableTable(tableName);
			admin.deleteTable(tableName);
		}
		System.out.println("Drop table " + tableName + "successfully");
	}
	public static void main(String[] args) {
		try {
			TableName tableName = TableName.valueOf("students");
			HBaseStudy.dropTable(tableName);
			
			String[] familys = {"Description", "Courses", "Home"};
			String[] desQualifiers = {"Name", "Height"};
			String[] CouQualifiers = {"Chinese", "Math", "Physics"};
			String[] HomQualifiers = {"Province"};
			String[][] Records = {{"Li Lei", "176", "80", "90", "95", "Zhejiang"}, 
					{"Han Meimei", "183", "88", "77", "66", "Beijing"},
					{"Xiao Ming", "162", "90", "90", "90", "Shanghai"} };
			HBaseStudy.createTable(tableName, familys);
			for(int i = 0; i < Records.length; i++){
				int index = i + 1;
				HBaseStudy.addData(tableName, "00" + index, familys[0], desQualifiers[0], Records[i][0]);
				HBaseStudy.addData(tableName, "00" + index, familys[0], desQualifiers[1], Records[i][1]);
				HBaseStudy.addData(tableName, "00" + index, familys[1], CouQualifiers[0], Records[i][2]);
				HBaseStudy.addData(tableName, "00" + index, familys[1], CouQualifiers[1], Records[i][3]);
				HBaseStudy.addData(tableName, "00" + index, familys[1], CouQualifiers[2], Records[i][4]);
				HBaseStudy.addData(tableName, "00" + index, familys[2], HomQualifiers[0], Records[i][5]);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}