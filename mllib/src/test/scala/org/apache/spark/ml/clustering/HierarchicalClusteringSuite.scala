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
package org.apache.spark.ml.clustering

import org.scalatest.FunSuite

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

private[spark]
case class TestPoint(point: Vector)

private[spark]
object HierarchicalClusteringSuite {

  def generateTestDataFrame( sqlContext: SQLContext, numClusters: Int, size: Int): DataFrame = {
    import sqlContext.implicits._

    val local = (0 to size).map { i =>
      val elms = Array.fill(size)(0.0)
      elms(i % numClusters) = i % numClusters
      Vectors.dense(elms)
    }.toArray

    val sc = sqlContext.sparkContext
    val rdd = sc.parallelize(local, 2)
    val points = rdd.map (new TestPoint(_))
    points.toDF("point")
  }

  def generateTestDataFrameWithSparceVectors(
    sqlContext: SQLContext,
    numClusters: Int,
    size: Int): DataFrame = {
    import sqlContext.implicits._

    val local = (0 to size).map { i =>
      val indexes = Array(i % numClusters)
      val values = Array((i % numClusters).toDouble)
      Vectors.sparse(numClusters, indexes, values)
    }.toArray

    val sc = sqlContext.sparkContext
    val rdd = sc.parallelize(local, 2)
    val points = rdd.map (new TestPoint(_))
    points.toDF("point")
  }
}

class HierarchicalClusteringSuite extends FunSuite with MLlibTestSparkContext with Logging {

  test("getter & setter") {
    val algo = new HierarchicalClustering()

    // test default values and getters
    intercept[NoSuchElementException] {
      algo.getNumClusters
    }
    assert(algo.getMaxIter === 20)
    assert(algo.getMaxRetries === 5)
    assert(algo.getSeed === 1)
    assert(algo.getFeaturesCol === "features")

    // test setters
    algo.setNumClusters(9)
    assert(algo.getNumClusters === 9)
    algo.setMaxIter(100)
    assert(algo.getMaxIter === 100)
    algo.setMaxRetries(50)
    assert(algo.getMaxRetries === 50)
    algo.setSeed(999)
    assert(algo.getSeed === 999)
    algo.setFeaturesCol("elements")
    assert(algo.getFeaturesCol === "elements")
  }

  test("fit & predict with dense vectors") {
    val sqlContext = this.sqlContext
    val algo = new HierarchicalClustering()
        .setNumClusters(5)
        .setFeaturesCol("point")
        .setMaxIter(20)
        .setMaxRetries(5)
        .setSeed(1)

    val dataset = HierarchicalClusteringSuite.generateTestDataFrame(sqlContext, 5, 99)
    val model = algo.fit(dataset)

    assert(model.getCenters.length === 5)
    assert(model.getClusters.length === 5)
    (0 to (model.getCenters.length - 1)).foreach { i =>
      val point = model.getCenters.apply(i)
      assert(model.predict(point) === i)
    }

    // convert into a linkage matrix
    val linkageMatrix = model.toLinkageMatrix()
    assert(linkageMatrix.length === 4)

    // convert into an adjacency list
    val adjacencyList = model.toAdjacencyList()
    assert(adjacencyList.length === 8)
  }

  test("fit & predict with sparse vectors") {
    val sqlContext = this.sqlContext
    val algo = new HierarchicalClustering()
        .setNumClusters(5)
        .setFeaturesCol("point")
        .setMaxIter(20)
        .setMaxRetries(5)
        .setSeed(1)
    val dataset =
      HierarchicalClusteringSuite.generateTestDataFrameWithSparceVectors(sqlContext, 5, 99)
    val model = algo.fit(dataset)

    assert(model.getCenters.length === 5)
    assert(model.getClusters.length === 5)
    (0 to (model.getCenters.length - 1)).foreach { i =>
      val point = model.getCenters.apply(i)
      assert(model.predict(point) === i)
    }

    // convert into a linkage matrix
    val linkageMatrix = model.toLinkageMatrix()
    assert(linkageMatrix.length === 4)

    // convert into an adjacency list
    val adjacencyList = model.toAdjacencyList()
    assert(adjacencyList.length === 8)
  }
}
