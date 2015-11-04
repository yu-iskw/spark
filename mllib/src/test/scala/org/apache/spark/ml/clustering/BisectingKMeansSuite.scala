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

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.clustering.{BisectingKMeans => MLlibKMeans}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.DataFrame

class BisectingKMeansSuite extends SparkFunSuite with MLlibTestSparkContext {

  final val k = 5
  @transient var dataset: DataFrame = _

  override def beforeAll(): Unit = {
    super.beforeAll()

    dataset = KMeansSuite.generateKMeansData(sqlContext, 50, 3, k)
  }

  test("default parameters") {
    val BisectingKMeans = new BisectingKMeans()

    assert(BisectingKMeans.getK === 2)
    assert(BisectingKMeans.getFeaturesCol === "features")
    assert(BisectingKMeans.getPredictionCol === "prediction")
    assert(BisectingKMeans.getMaxIter === 20)
  }

  test("set parameters") {
    val BisectingKMeans = new BisectingKMeans()
      .setK(9)
      .setFeaturesCol("test_feature")
      .setPredictionCol("test_prediction")
      .setMaxIter(33)
      .setSeed(123)

    assert(BisectingKMeans.getK === 9)
    assert(BisectingKMeans.getFeaturesCol === "test_feature")
    assert(BisectingKMeans.getPredictionCol === "test_prediction")
    assert(BisectingKMeans.getMaxIter === 33)
    assert(BisectingKMeans.getSeed === 123)
  }

  test("parameters validation") {
    intercept[IllegalArgumentException] {
      new BisectingKMeans().setK(1)
    }
  }

  test("fit & transform") {
    val predictionColName = "bisecting_kmeans_prediction"
    val BisectingKMeans = new BisectingKMeans().setK(k).setPredictionCol(predictionColName).setSeed(1)
    val model = BisectingKMeans.fit(dataset)
    assert(model.getCenters.length === k)

    val transformed = model.transform(dataset)
    val expectedColumns = Array("features", predictionColName)
    expectedColumns.foreach { column =>
      assert(transformed.columns.contains(column))
    }
    val clusters = transformed.select(predictionColName).map(_.getInt(0)).distinct().collect().toSet
    assert(clusters.size === k)
    assert(clusters === Set(0, 1, 2, 3, 4))
    assert(model.computeCost(dataset) < 0.1)
  }
}
