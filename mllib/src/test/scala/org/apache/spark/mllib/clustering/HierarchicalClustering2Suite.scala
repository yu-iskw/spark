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

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm => breezeNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.scalatest.FunSuite

class HierarchicalClustering2Suite
    extends FunSuite with MLlibTestSparkContext {

  test("run") {
    val algo = new HierarchicalClustering2
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val seed = sc.parallelize(localSeed)
    algo.run(seed)
  }

  test("initializeData") {
    val algo = new HierarchicalClustering2
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val seed = sc.parallelize(localSeed)
    val data = algo.initializeData(seed)
    assert(data.map(_._1).collect().distinct === Array(1))
  }

  test("get center stats") {
    val algo = new HierarchicalClustering2
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val seed = sc.parallelize(localSeed)
    val data = algo.initializeData(seed)

    val clusters = algo.getCenterStats(data)
    val center = clusters(1).center
    assert(clusters.size === 1)
    assert(clusters(1).center === Vectors.dense(49.5, 49.5))
    assert(clusters(1).records === 100)

    val data2 = seed.map(v => ((v.apply(0) / 25).toInt + 1, v.toBreeze))
    val clusters2 = algo.getCenterStats(data2)
    assert(clusters2.size === 4)
    assert(clusters2(1).center === Vectors.dense(12.0, 12.0))
    assert(clusters2(1).records === 25)
    assert(clusters2(2).center === Vectors.dense(37.0, 37.0))
    assert(clusters2(2).records === 25)
    assert(clusters2(3).center === Vectors.dense(62.0, 62.0))
    assert(clusters2(3).records === 25)
    assert(clusters2(4).center === Vectors.dense(87.0, 87.0))
    assert(clusters2(4).records === 25)
  }

  test("takeInitCenter: the relative error should be equal to or less than 0.1") {
    val algo = new HierarchicalClustering2
    val center = Vectors.dense(1.0, 2.0, 3.0).toBreeze
    val nextCenters = algo.takeInitCenters(center)
    nextCenters.foreach { vector =>
      val error = (center - vector) :/ center
      assert(error.values.forall(_ <= 0.1))
    }
  }

  test("split") {
    val algo = new HierarchicalClustering2
    val seed = (0 to 99).map(i => ((i / 50).toInt + 2, Vectors.dense(i, i).toBreeze))
    val data = sc.parallelize(seed)
    val newClusters = algo.splitCenters(data)

    assert(newClusters.size === 4)
    assert(newClusters(4).center === Vectors.dense(12.0, 12.0))
    assert(newClusters(4).records === 25)
    assert(newClusters(5).center === Vectors.dense(37.0, 37.0))
    assert(newClusters(5).records === 25)
    assert(newClusters(6).center === Vectors.dense(62.0, 62.0))
    assert(newClusters(6).records === 25)
    assert(newClusters(7).center === Vectors.dense(87.0, 87.0))
    assert(newClusters(7).records === 25)
  }
}
