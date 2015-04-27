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

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.SQLContext
import org.scalatest.FunSuite

case class TestDataFrame(point: Vector)

class HierarchicalClusteringSuite extends FunSuite with MLlibTestSparkContext with Logging {

  private var sqlContext: SQLContext = _

  override def beforeAll(): Unit = {
    super.beforeAll()
    sqlContext = new SQLContext(sc)
  }

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
    algo.setInputCol("elements")
    assert(algo.getFeaturesCol === "elements")
  }

  test("fit & predict") {
    val sqlContext = this.sqlContext
    // this is used to implicitly convert an RDD to a DataFrame.
    import sqlContext.implicits._

    val local = (0 to 99).map { i =>
      val elm = (i % 5).toDouble
      val point = Vectors.dense(elm, elm, elm)
      TestDataFrame(point)
    }
    val dataset = sc.makeRDD(local, 2).toDF("point")

    val algo = new HierarchicalClustering()
        .setNumClusters(5)
        .setInputCol("point")
        .setMaxIter(20)
        .setMaxRetries(5)
        .setSeed(1)
    val model = algo.fit(dataset)

    assert(model.predict(local(0).point) === model.predict(local(5).point))
    assert(model.predict(local(1).point) === model.predict(local(6).point))
    assert(model.predict(local(2).point) === model.predict(local(7).point))
    assert(model.predict(local(3).point) === model.predict(local(8).point))
    assert(model.predict(local(4).point) === model.predict(local(9).point))

    assert(model.predict(local(0).point) !== model.predict(local(1).point))
    assert(model.predict(local(0).point) !== model.predict(local(2).point))
    assert(model.predict(local(0).point) !== model.predict(local(3).point))
    assert(model.predict(local(0).point) !== model.predict(local(4).point))
  }
}
