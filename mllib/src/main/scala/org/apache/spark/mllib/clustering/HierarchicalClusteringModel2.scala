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

import breeze.linalg.{DenseVector => BDV, Vector => BV, norm => breezeNorm}
import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD

class HierarchicalClusteringModel2(val tree: ClusterTree2) extends Serializable with Logging {

  def getClusters(): Array[ClusterTree2] = this.tree.getLeavesNodes()

  def getCenters(): Array[Vector] = this.getClusters().map(_.center)

  /**
   * Predicts the closest cluster of each point
   */
  def predict(vector: Vector): Int = {
    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)

    val centers = this.getCenters().map(_.toBreeze)
    HierarchicalClustering2.findClosestCenter(metric)(centers)(vector.toBreeze)
  }

  /**
   * Predicts the closest cluster of each point
   */
  def predict(data: RDD[Vector]): RDD[Int] = {
    val sc = data.sparkContext

    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)
    sc.broadcast(metric)
    val centers = this.getCenters().map(_.toBreeze)
    sc.broadcast(centers)

    data.map(point => HierarchicalClustering2.findClosestCenter(metric)(centers)(point.toBreeze))
  }
}

