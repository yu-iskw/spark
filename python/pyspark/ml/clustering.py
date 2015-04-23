#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark.ml.util import keyword_only
from pyspark.ml.wrapper import JavaEstimator, JavaModel
from pyspark.ml.param.shared import *
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter
from pyspark.mllib.common import inherit_doc

__all__ = ['HierarchicalClustering', 'HierarchicalClusteringModel']


class HierarchicalClusteringModel(JavaModel):
    """
    Model fitted by Hierarchical Clustering
    """

    def getCenters(self):
        """Get the cluster centers, represented as a list of NumPy arrays."""
        return [c.toArray() for c in self._call_java("getCenters")]

    def toAdjacencyList(self):
        """Converts to an adjacency list."""
        return self._call_java("toJavaAdjacencyList")

    def toLinkageMatrix(self):
        """Converts to a linkage matrix"""
        return self._call_java("toJavaLinkageMatrix")


@inherit_doc
class HierarchicalClustering(JavaEstimator, HasFeaturesCol, HasSeed, HasMaxIter):
    """
    Hierarchical Clustering

    >>> from pyspark.mllib.linalg import Vectors
    >>> from numpy.testing import assert_almost_equal
    >>> data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
    ...         (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
    >>> df = sqlContext.createDataFrame(data, ["point"])
    >>> hc = HierarchicalClustering().setNumClusters(2).setSeed(1).setFeaturesCol("point")
    >>> model = hc.fit(df)
    >>> centers = model.getCenters()
    >>> len(centers)
    2
    >>> transformed = model.transform(df)
    >>> (transformed.columns)[0] == 'point'
    True
    >>> (transformed.columns)[1] == 'prediction'
    True
    >>> rows = sorted(transformed.collect(), key = lambda r: r[0])
    >>> rows[0].prediction == rows[1].prediction
    True
    >>> rows[2].prediction == rows[3].prediction
    True
    >>> expected = [[0.0, 1.0, 5.6], [0.0, 2.0, 5.6]]
    >>> assert_almost_equal(model.toAdjacencyList(), expected, 1)
    >>> expected2 = [[0.0, 1.0, 5.6, 2.0]]
    >>> assert_almost_equal(model.toLinkageMatrix(), expected2, 1)
    """
    _java_class = "org.apache.spark.ml.clustering.HierarchicalClustering"

    @keyword_only
    def __init__(self, numClusters=2, maxRetries=5):
        super(HierarchicalClustering, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.clustering.HierarchicalClustering", self.uid)
        self.numClusters = Param(self, "numClusters", "number of clusters you want")
        self.maxRetries = Param(self, "maxRetries", "maximum number of retries")
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)

    def _create_model(self, java_model):
        return HierarchicalClusteringModel(java_model)

    @keyword_only
    def setParams(self, numClusters=2, maxRetries=5):
        kwargs = self.setParams._input_kwargs
        return self._set(**kwargs)

    def setNumClusters(self, value):
        """
        Sets the value of :py:attr:`numClusters`.

        >>> algo = HierarchicalClustering().setNumClusters(123)
        >>> algo.getNumClusters()
        123
        """
        self._paramMap[self.numClusters] = value
        return self

    def getNumClusters(self):
        """
        Gets the value of `numClusters`
        """
        return self.getOrDefault(self.numClusters)

    def setMaxRetries(self, value):
        """
        Sets the value of :py:attr:`maxRetries`.

        >>> algo = HierarchicalClustering().setMaxRetries(123)
        >>> algo.getMaxRetries()
        123
        """
        self._paramMap[self.maxRetries] = value
        return self

    def getMaxRetries(self):
        """
        Gets the value of `maxRetries`
        """
        return self.getOrDefault(self.maxRetries)


if __name__ == "__main__":
    import doctest
    from pyspark.context import SparkContext
    from pyspark.sql import SQLContext
    globs = globals().copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    sc = SparkContext("local[2]", "ml.clustering tests")
    sqlContext = SQLContext(sc)
    globs['sc'] = sc
    globs['sqlContext'] = sqlContext
    (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
    sc.stop()
    if failure_count:
        exit(-1)
