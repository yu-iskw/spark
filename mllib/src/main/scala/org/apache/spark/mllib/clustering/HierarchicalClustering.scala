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
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

class HierarchicalClusteringModel extends Serializable {
}

class HierarchicalClustering private (
  private var L: Int,
  private var w: Int
) extends Serializable with Logging {

  var seed: Long = _

  def run(data: RDD[Vector]): HierarchicalClusteringModel = {
    new HierarchicalClusteringModel
  }

  def setHashes(L: Integer): this.type = {
    this.L = L
    this
  }

  def setQuantificationLevels(w: Integer): this.type = {
    this.w = w
    this
  }

  def setSeed(seed: Long): this.type = {
    this.seed = seed
    this
  }
}

/**
 * Top-level methods for calling Hierarchical Clustering
 */
object HierarchicalClustering {
  val DEFAULT_SEED = 12345

  def train(L: Integer, w: Integer, seed: Long = DEFAULT_SEED): HierarchicalClustering = {
    new HierarchicalClustering(L, w)
      .setSeed(seed)
  }
}
