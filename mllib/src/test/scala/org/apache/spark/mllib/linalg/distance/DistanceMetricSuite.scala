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

import breeze.linalg.{Vector => BV}
import org.apache.spark.mllib.linalg.Vectors

class ChebyshevDistanceMetricSuite extends GeneralDistanceMetricSuite {
  override def distanceFactory: DistanceMetric = new ChebyshevDistanceMetric

  test("the distance should be 6") {
    val vector1 = Vectors.dense(1, -1, 1, -1).toBreeze
    val vector2 = Vectors.dense(2, -3, 4, 5).toBreeze
    val distance = distanceFactory(vector1, vector2)
    assert(distance == 6, s"the distance should be 6, actual ${distance}")
  }

  test("the distance should be 100") {
    val vector1 = Vectors.dense(1, -1, 1, -1).toBreeze
    val vector2 = Vectors.dense(101, -3, 4, 5).toBreeze
    val distance = distanceFactory(vector1, vector2)
    assert(distance == 100, s"the distance should be 100, actual ${distance}")
  }
}

class EuclideanDistanceMetricSuite extends GeneralDistanceMetricSuite {
  override def distanceFactory = new EuclideanDistanceMetric

  test("the distance should be 6.7082039325") {
    val v1 = Vectors.dense(2, 3).toBreeze
    val v2 = Vectors.dense(5, 9).toBreeze

    val distance = distanceFactory(v1, v2)
    val expected = 6.7082039325
    val isNear = GeneralDistanceMetricSuite.isNearlyEqual(distance, expected)
    assert(isNear, s"the distance should be nearly equal to ${expected}, actual ${distance}")
  }
}

class ManhattanDistanceMetricSuite extends GeneralDistanceMetricSuite {
  override def distanceFactory = new ManhattanDistanceMetric()

  test("the distance should be 6.7082039325") {
    val v1 = Vectors.dense(2, 3, 6, 8).toBreeze
    val v2 = Vectors.dense(5, 9, 1, 4).toBreeze

    val distance = distanceFactory(v1, v2)
    val expected = 18.0
    val isNear = GeneralDistanceMetricSuite.isNearlyEqual(distance, expected)
    assert(isNear, s"the distance should be nearly equal to ${expected}, actual ${distance}")
  }
}

class MinkowskiDistanceMetricSuite extends GeneralDistanceMetricSuite {
  override def distanceFactory: DistanceMetric = new MinkowskiDistanceMetric(4.0)

  test("the distance between the vectors should be expected") {
    val vector1 = Vectors.dense(0, 0, 0).toBreeze
    val vector2 = Vectors.dense(2, 3, 4).toBreeze

    val measure = new MinkowskiDistanceMetric(3.0)
    val distance = measure(vector1, vector2)
    val expected = 4.6260650092
    val isNear = GeneralDistanceMetricSuite.isNearlyEqual(distance, expected)
    assert(isNear, s"the distance between the vectors should be ${expected}")
  }
}
