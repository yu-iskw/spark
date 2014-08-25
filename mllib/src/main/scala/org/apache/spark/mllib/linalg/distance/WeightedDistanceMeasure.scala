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

import breeze.linalg.sum
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.Vector

/**
 * this trait is used for a weighted distance measure
 */
@Experimental
trait WeightedDistanceMeasure extends DistanceMeasure {
  val weights: Vector
}


/**
 * A weighted Cosined distance measure implementation
 *
 * (sum w[i]*u[i]*v[i]) / sqrt[(sum w[i]*u[i]^2)*(sum w[i]*v[i]^2)].
 *
 * @param weights weight vector
 */
@Experimental
class WeightedCosineDistanceMeasure(val weights: Vector) extends WeightedDistanceMeasure {

  override def apply(v1: Vector, v2: Vector): Double = {
    val wbv = weights.toBreeze
    val bv1 = v1.toBreeze
    val bv2 = v2.toBreeze

    val dotProduct = sum(wbv :* (bv1 :* bv2))
    var denominator = Math.sqrt(sum(wbv :* bv1 :* bv1) * sum(wbv :* bv2 :* bv2))

    // correct for floating-point rounding errors
    if (denominator < dotProduct) {
      denominator = dotProduct
    }

    // correct for zero-vector corner case
    if (denominator == 0 && dotProduct == 0) {
      return 0.0
    }
    1.0 - (dotProduct / denominator)
  }
}
