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



if __name__ == '__main__':
    InsideCircleDataGenerator().gen_file("../data/circles.tsv", 100, 2, 10, 6)