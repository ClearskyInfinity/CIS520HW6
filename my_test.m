a = [1, 2];
b = [3, 4];
c = [5, 6];

buckets = {a, b, c};

t = cat(1, buckets{:});

disp(sum(t, 1));