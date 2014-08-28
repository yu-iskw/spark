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

import breeze.linalg.{DenseVector => DBV, Vector => BV, sum, max}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.Vector

/**
 * :: Experimental ::
 * This trait is used for objects which can determine a distance metric between two points
 *
 * 1. d(x, y) >= 0 (non-negative)
 * 2. d(x, y) = 0 if and only if x = y (identity of indiscernibles)
 * 3. d(x, y) = d(y, x) (symmetry)
 * 4. d(x, z) <= d(x, y) + d(y, z) (triangle inequality)
 */
@Experimental
trait DistanceMetric extends DistanceMeasure

/**
 * :: Experimental ::
 * Euclidean distance implementation
 */
@Experimental
class EuclideanDistanceMetric extends DistanceMetric {

  /**
   * Calculates the euclidean distance (L2 distance) between 2 points
   *
   * D(x, y) = sqrt(sum((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2))
   * or
   * D(x, y) = sqrt((x-y) dot (x-y))
   *
   * @param v1
   * @param v2
   * @return
   */
  override def apply(v1: Vector, v2: Vector): Double = {
    val d = v1.toBreeze - v2.toBreeze
    Math.sqrt(d dot d)
  }
}

/**
 * :: Experimental ::
 * A weighted Euclidean distance metric implementation
 * this metric is calculated by summing the square root of the squared differences
 * between each coordinate, optionally adding weights.
 */
@Experimental
class WeightedEuclideanDistanceMetric(val weights: Vector) extends DistanceMetric {

  override def apply(v1: Vector, v2: Vector): Double = {
    val d = v1.toBreeze - v2.toBreeze
    Math.sqrt(d dot (weights.toBreeze :* d))
  }
}


/**
 * :: Experimental ::
 * Chebyshev distance implementation
 *
 * @see http://en.wikipedia.org/wiki/Chebyshev_distance
 */
@Experimental
class ChebyshevDistanceMetric extends DistanceMetric {

  /**
   * Calculates a Chebyshev distance metric
   *
   * d(a, b) := max{|a(i) - b(i)|} for all i
   *
   * @param v1 a Vector defining a multidimensional point in some feature space
   * @param v2 a Vector defining a multidimensional point in some feature space
   * @return Double a distance
   */
  override def apply(v1: Vector, v2: Vector): Double = {
    val diff = (v1.toBreeze - v2.toBreeze).map(Math.abs)
    max(diff)
  }
}

/**
 * :: Experimental ::
 * A weighted Chebyshev distance implementation
 */
@Experimental
class WeightedChebyshevDistanceMetric(val weights: Vector) extends DistanceMetric {

  /**
   * Calculates a weighted Chebyshev distance metric
   *
   * d(a, b) := max{w(i) * |a(i) - b(i)|} for all i
   * where w is a weighted vector
   *
   * @param v1 a Vector defining a multidimensional point in some feature space
   * @param v2 a Vector defining a multidimensional point in some feature space
   * @return Double a distance
   */
  override def apply(v1: Vector, v2: Vector): Double = {
    val diff = (v1.toBreeze - v2.toBreeze).map(Math.abs).:*(weights.toBreeze)
    max(diff)
  }
}

/**
 * :: Experimental ::
 * Manhattan distance (L1 distance) implementation
 *
 * @see http://en.wikipedia.org/wiki/Manhattan_distance
 */
@Experimental
class ManhattanDistanceMetric extends DistanceMetric {

  override def apply(v1: Vector, v2: Vector): Double = {
    sum((v1.toBreeze - v2.toBreeze).map(Math.abs))
  }
}

/**
 * :: Experimental ::
 * A weighted Manhattan distance metric implementation
 * this metric is calculated by summing the absolute values of the difference
 * between each coordinate, optionally with weights.
 */
@Experimental
class WeightedManhattanDistanceMetric(val weights: Vector) extends DistanceMetric {

  override def apply(v1: Vector, v2: Vector): Double = {
    weights.toBreeze dot ((v1.toBreeze - v2.toBreeze).map(Math.abs))
  }
}

/**
 * Minkowski distance Implementation
 * The Minkowski distance is a metric on Euclidean space
 * which can be considered as a generalization of both
 * the Euclidean distance and the Manhattan distance.
 *
 * @see http://en.wikipedia.org/wiki/Minkowski_distance
 */
@Experimental
class MinkowskiDistanceMetric(val exponent: Double) extends DistanceMetric {

  // the default value for exponent
  def this() = this(3.0)

  override def apply(v1: Vector, v2: Vector): Double = {
    val d = (v1.toBreeze - v2.toBreeze).map(diff => Math.pow(Math.abs(diff), exponent))
    Math.pow(sum(d), 1 / exponent)
  }
}
