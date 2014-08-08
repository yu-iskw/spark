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

package org.apache.spark.mllib.hash.lsh

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.scalatest.{FunSuite, ShouldMatchers}


class RandomProjectionHashingSuite extends FunSuite with ShouldMatchers {
  test("generate a random vector") {
    val elements = 5
    val value = RandomProjectionHashing.getRandomGaussianVector(elements)
    value.size should be(5)
  }

  test("generate a random vector with seed") {
    val elements = 5
    val seed = 12345
    val value = RandomProjectionHashing.getRandomGaussianVector(elements, seed)
    val expected = Vectors.dense(
      Array(-0.187808989658912, 0.5884363051154796, 0.9488047804400426, -0.49428072062604445, -1.223411937180115)
    )
    value.size should be(5)
    value should be(expected)
  }

  test("hash") {
    val vectors = Array(
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.0, 0.1),
      Vectors.dense(0.1, 0.0),

      Vectors.dense(5.0, 5.0),
      Vectors.dense(5.0, 5.1),
      Vectors.dense(5.1, 5.0),

      Vectors.dense(10.0, 10.0),
      Vectors.dense(9.9, 10.0),
      Vectors.dense(10.0, 9.9),
      Vectors.dense(9.9, 9.9)
    )

    val expected = Array(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0)

    val w = 2
    val b = 1
    val x = RandomProjectionHashing.getRandomGaussianVector(vectors(0).size)

    vectors.zipWithIndex.foreach { case (v: Vector, i: Int) =>
      val d = v.size
      val result = RandomProjectionHashing.hash(v, x, w, b)
      print(result.toString)
    }

  }

}
