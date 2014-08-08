/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.clustering

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.feature.{EuclideanLSHTransformer, EuclideanLSH}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

@Experimental
class HierarchicalClusteringModel extends Serializable {
}

@Experimental
class HierarchicalClustering private extends Serializable with Logging {

  def run(data: RDD[Vector], L: Int, w: Int): HierarchicalClusteringModel = {
    val dimension = data.first().size
    val randomVector = EuclideanLSH.generateRandomVectors(L, data.first().size)
    val vectorTransformer = new EuclideanLSHTransformer(dimension, L, w)
    new HierarchicalClusteringModel
  }
}

/**
 * Top-level methods for calling Hierarchical Clustering
 */
@Experimental
object HierarchicalClustering {

  def train(data: RDD[Vector], L: Int, w: Int): HierarchicalClusteringModel = {
    new HierarchicalClustering()
      .run(data, L, w)
  }

}
