import numpy as np
import numpy.random as npr

class InsideCircleDataGenerator:
    def generate(self, count, dim_n, max_value, r):
        data = npr.rand(count, dim_n) * 2 * max_value - max_value
        squared_distances = (data**2).sum(1)
        labels = np.vectorize(int)(squared_distances < r*r)
        labels = np.matrix(labels).transpose()
        return np.hstack((data, labels))

    def gen_file(self, fname, count, dim_n, max_value, r):
        data = self.generate(count, dim_n, max_value, r)
        str_data = np.vectorize(lambda x: '%.3f' % x)(data)
        with open(fname, 'w') as f:
            f.write('%d %d\n' % (dim_n, 1))
            for sample in str_data:
                f.write('\t'.join(sample.flat) + '\n')

class InsideRingDataGenerator:
    def generate(self, count, dim_n, max_value, r1, r2):
        data = npr.rand(count, dim_n) * 2 * max_value - max_value
        squared_distances = (data**2).sum(1)
        labels = np.vectorize(int)(np.logical_and(squared_distances < r1*r1, squared_distances > r2*r2))
        labels = np.matrix(labels).transpose()
        return np.hstack((data, labels))

    def gen_file(self, fname, count, dim_n, max_value, r1, r2):
        data = self.generate(count, dim_n, max_value, r1, r2)
        str_data = np.vectorize(lambda x: '%.3f' % x)(data)
        with open(fname, 'w') as f:
            f.write('%d %d\n' % (dim_n, 1))
            for sample in str_data:
                f.write('\t'.join(sample.flat) + '\n')

if __name__ == '__main__':
    InsideRingDataGenerator().gen_file("../data/rings.tsv", 1000, 2, 10, 6, 3)