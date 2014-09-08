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

package org.apache.spark.mllib.linalg.distance

import org.apache.spark.mllib.linalg.Vectors

class WeightedCosineDistanceMeasureSuite extends GeneralDistanceMeasureSuite {

  override def distanceFactory: DistanceMeasure = {
    val weights = Vectors.dense(0.1, 0.1, 0.1, 0.1, 0.1, 0.5) // size should be 6
    new WeightedCosineDistanceMeasure(weights)
  }

  test("concreate distance check") {
    val vector1 = Vectors.dense(1.0, 2.0)
    val vector2 = Vectors.dense(3.0, 4.0)

    val measure = new WeightedCosineDistanceMeasure(Vectors.dense(0.5, 0.5))
    val distance = measure(vector1, vector2)
    val expected = 0.01613008990009257
    val isNear = GeneralDistanceMetricSuite.isNearlyEqual(distance, expected)
    assert(isNear, s"the distance should be nearly equal to ${expected}, actual ${distance}")
  }

  test("two vectors have the same magnitude") {
    val vector1 = Vectors.dense(1.0, 1.0)
    val vector2 = Vectors.dense(2.0, 2.0)

    val measure = new WeightedCosineDistanceMeasure(Vectors.dense(0.5, 0.5))
    val distance = measure(vector1, vector2)
    val expected = 0.0
    val isNear = GeneralDistanceMetricSuite.isNearlyEqual(distance, expected)
    assert(isNear, s"the distance should be nearly equal to ${expected}, actual ${distance}")
  }
}

