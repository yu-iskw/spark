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

package org.apache.spark.mllib.feature

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.util.random.XORShiftRandom

import scala.util.Random

@Experimental
class EuclideanLSH extends Serializable {

  def calculateHashes(randomVectors: Array[Vector])(w: Int, b: Int)(vector: Vector): Vector = {
    val hashes = randomVectors.map(randomVector => this.calculate(randomVector, w, b)(vector))
    Vectors.dense(hashes)
  }

  def calculate(randomVector: Vector, w: Int, b: Int)(vector: Vector): Double = {
    val dotProduct = vector.toBreeze.dot(randomVector.toBreeze)
    Math.floor((dotProduct + b) / w)
  }

}

@Experimental
object EuclideanLSH {

  def generateRandomVectors(L: Int, dimension: Int): Array[Vector] = {
    (1 to L).toArray.map { _ =>
      val seed = new XORShiftRandom().nextInt()
      generateRandomVector(dimension)
    }
  }

  def generateRandomVector(dimension: Int): Vector = {
    val seed = new XORShiftRandom().nextInt()
    generateRandomVector(dimension, seed)
  }

  def generateRandomVector(dimension: Int, seed: Int): Vector = {
    val random = new Random(seed.toLong)
    Vectors.dense((1 to dimension).map(_ => random.nextGaussian()).toArray)
  }
}

