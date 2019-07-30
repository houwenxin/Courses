public class UnionFind {
    public static class QuickFindUF {
        private int[] ids;
        public QuickFindUF(int N) {
            ids = new int[N];
            for(int i = 0; i < N; i++){
                ids[i] = i;
            }
            for(int id : ids) {
                System.out.println(id);
            }
        }
        
    }
    public static void main(String[] args)
    {
        QuickFindUF uf = new QuickFindUF(N);
    }
}
