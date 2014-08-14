/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.distance

import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.scalatest.{FunSuite, ShouldMatchers}

private[distance]
abstract class GeneralDistanceMeasureSuite extends FunSuite with ShouldMatchers {
  val distanceMeasureFactory: DistanceMeasure

  val vectors = Array(
    Vectors.dense(1, 1, 1, 1, 1, 1),
    Vectors.dense(2, 2, 2, 2, 2, 2),
    Vectors.dense(6, 6, 6, 6, 6, 6),
    Vectors.dense(-1, -1, -1, -1, -1, -1)
  )


  test("measure the distances between two vector") {
    GeneralDistanceMeasureSuite.compare(distanceMeasureFactory, vectors)
  }
}

object GeneralDistanceMeasureSuite {

  def compare(distanceMeasure: DistanceMeasure, vectors: Array[Vector]) {
    val denseMatrixElements = for (v1 <- vectors; v2 <- vectors) yield {
      distanceMeasure.distance(v2, v1)
    }

    val distanceMatrix = Matrices.dense(vectors.size, vectors.size, denseMatrixElements)

    assert(distanceMatrix(0, 0) < distanceMatrix(0, 1))
    assert(distanceMatrix(0, 1) < distanceMatrix(0, 2))

    assert(distanceMatrix(1, 0) > distanceMatrix(1, 1))
    assert(distanceMatrix(1, 2) > distanceMatrix(1, 0))

    assert(distanceMatrix(2, 0) > distanceMatrix(2, 1))
    assert(distanceMatrix(2, 1) > distanceMatrix(2, 2))

    for (i <- 0 to (vectors.size - 1); j <- 0 to (vectors.size - 1)) {
      i match {
        // diagonal element in the distance matrix
        case j => {
          assert(distanceMatrix(i, i) == 0.0, "Diagonal elements in the distance matrix is equal to zero")
        }
        // not diagnola element in the distance matrix
        case _ => {
          assert(distanceMatrix(i, j) > 0, "Distance between vectors greater than zero")
        }
      }
    }
  }

}
