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
from pyspark.ml.param.shared import HasFeaturesCol, HasMaxIter
from pyspark.mllib.common import inherit_doc

__all__ = ['LogisticRegression', 'LogisticRegressionModel']

@inherit_doc
class HierarchicalClustering(JavaEstimator, HasFeaturesCol, HasMaxIter):
    """
    Hierarchical Clustering

    >>> algo = HierarchicalClustering(features="point", maxIter=10)
    """
    _java_class = "org.apache.spark.ml.clustering.HierarchicalClustering"

    @keyword_only
    def __init__(self, featuresCol="features", maxIter=100):
        super(LogisticRegression, self).__init__()
        kwargs = self.__init__._input_kwargs
        self.setParams(**kwargs)



class HierarchicalClusteringModel():
    """
    Model fitted by Hierarchical Clustering
    """


if __name__ == "__main__":
    import doctest
    from pyspark.context import SparkContext
    from pyspark.sql import SQLContext
    globs = globals().copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    sc = SparkContext("local[2]", "ml.clustering tests")
    sqlCtx = SQLContext(sc)
    globs['sc'] = sc
    globs['sqlCtx'] = sqlCtx
    (failure_count, test_count) = doctest.testmod(
        globs=globs, optionflags=doctest.ELLIPSIS)
    sc.stop()
    if failure_count:
        exit(-1)
