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


class HierarchicalClustering2AppSuite extends FunSuite with MLlibTestSparkContext {

  test("train") {
    val numClusters = 9
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val data = sc.parallelize(localSeed, 1)
    val model = HierarchicalClustering2.train(data, numClusters)
    assert(model.getClusters().size === numClusters)
  }

  test("train with full arguments") {
    val numClusters = 9
    val subIterations = 20
    val maxRetries = 20
    val seed = 321

    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val data = sc.parallelize(localSeed, 1)

    val model = HierarchicalClustering2.train(data, numClusters, subIterations, maxRetries, seed)
    assert(model.getClusters().size === numClusters)
  }
}

class HierarchicalClustering2Suite extends FunSuite with MLlibTestSparkContext {

  test("run") {
    val algo = new HierarchicalClustering2().setNumClusters(321)
    val localSeed: Seq[Vector] = (0 to 999).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val data = sc.parallelize(localSeed, 2)
    val model = algo.run(data)
    assert(model.tree.getLeavesNodes().size == 321)
  }

  test("run with too many cluster size than the records") {
    val algo = new HierarchicalClustering2().setNumClusters(123)
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val data = sc.parallelize(localSeed, 2)
    val model = algo.run(data)
    assert(model.tree.getLeavesNodes().size == 100)
  }

  test("initializeData") {
    val algo = new HierarchicalClustering2
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val seed = sc.parallelize(localSeed)
    val data = algo.initData(seed)
    assert(data.map(_._1).collect().distinct === Array(1))
  }

  test("get center stats") {
    val algo = new HierarchicalClustering2
    val localSeed: Seq[Vector] = (0 to 99).map(i => Vectors.dense(i.toDouble, i.toDouble)).toSeq
    val seed = sc.parallelize(localSeed)
    val data = algo.initData(seed)

    val clusters = algo.summarizeAsClusters(data)
    val center = clusters(1).center
    assert(clusters.size === 1)
    assert(clusters(1).center === Vectors.dense(49.5, 49.5))
    assert(clusters(1).records === 100)

    val data2 = seed.map(v => ((v.apply(0) / 25).toInt + 1, v.toBreeze))
    val clusters2 = algo.summarizeAsClusters(data2)
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

  test("getChildrenCenter") {
    val algo = new HierarchicalClustering2
    val localData = (1 to 99).map(i => (2, Vectors.dense(1.0).toBreeze)) ++
        (1 to 99).map(i => (3, Vectors.dense(1.0).toBreeze))
    val data = sc.parallelize(localData)
    val centers = algo.initChildrenCenter(data)
    assert(centers.size === 4)
    assert(centers.keySet === Set(4, 5, 6, 7))
  }

  test("split") {
    val algo = new HierarchicalClustering2
    val seed = (0 to 99).map(i => ((i / 50).toInt + 2, Vectors.dense(i, i).toBreeze))
    val data = sc.parallelize(seed)
    val clusters = algo.summarizeAsClusters(data)
    val newClusters = algo.getDivideClusters(data, clusters)

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

  test("assignToNewCluster") {
    val algo = new HierarchicalClustering2
    val seed = Seq(
      (2, Vectors.dense(0.0, 0.0)), (2, Vectors.dense(1.0, 1.0)), (2, Vectors.dense(2.0, 2.0)),
      (2, Vectors.dense(3.0, 3.0)), (2, Vectors.dense(4.0, 4.0)), (2, Vectors.dense(5.0, 5.0)),
      (3, Vectors.dense(6.0, 6.0)), (3, Vectors.dense(7.0, 7.0)), (3, Vectors.dense(8.0, 8.0)),
      (3, Vectors.dense(9.0, 9.0)), (3, Vectors.dense(10.0, 10.0)), (3, Vectors.dense(11.0, 11.0))
    ).map { case (idx, vector) => (idx, vector.toBreeze)}
    val newClusters = Map(
      4 -> new ClusterTree2(Vectors.dense(1.0, 1.0), 3, Vectors.dense(1.0, 1.0)),
      5 -> new ClusterTree2(Vectors.dense(4.0, 4.0), 3, Vectors.dense(1.0, 1.0)),
      6 -> new ClusterTree2(Vectors.dense(7.0, 7.0), 3, Vectors.dense(1.0, 1.0)),
      7 -> new ClusterTree2(Vectors.dense(10.0, 10.0), 3, Vectors.dense(1.0, 1.0))
    )
    val data = sc.parallelize(seed)
    val result = algo.updateClusterIndex(data, newClusters).collect().toSeq

    val expected = Seq(
      (4, Vectors.dense(0.0, 0.0)), (4, Vectors.dense(1.0, 1.0)), (4, Vectors.dense(2.0, 2.0)),
      (5, Vectors.dense(3.0, 3.0)), (5, Vectors.dense(4.0, 4.0)), (5, Vectors.dense(5.0, 5.0)),
      (6, Vectors.dense(6.0, 6.0)), (6, Vectors.dense(7.0, 7.0)), (6, Vectors.dense(8.0, 8.0)),
      (7, Vectors.dense(9.0, 9.0)), (7, Vectors.dense(10.0, 10.0)), (7, Vectors.dense(11.0, 11.0))
    ).map { case (idx, vector) => (idx, vector.toBreeze)}
    assert(result === expected)
  }

  test("setSubIterations") {
    val algo = new HierarchicalClustering2()
    assert(algo.getSubIterations() == 20)
    algo.setSubIterations(15)
    assert(algo.getSubIterations() == 15)
  }

  test("setNumRetries") {
    val algo = new HierarchicalClustering2()
    assert(algo.getMaxRetries() == 10)
    algo.setMaxRetries(15)
    assert(algo.getMaxRetries() == 15)
  }

  test("setSeed") {
    val algo = new HierarchicalClustering2()
    assert(algo.getSeed() == 1)
    algo.setSeed(987)
    assert(algo.getSeed() == 987)
  }
}
