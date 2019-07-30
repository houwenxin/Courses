function [distance, index, time] = WithVLFeat(testdata, dataset)
	kdtree = vl_kdtreebuild(dataset,'NumTrees', 1);
	t1 = clock;
	[distance, index] = vl_kdtreequery(kdtree, dataset, testdata, 'NumNeighbors', 1000, 'MaxComparisons', 6000);
	distance = distance(1);
	index = index(1);
	t2 = clock;
	time = etime(t2,t1);
	end
