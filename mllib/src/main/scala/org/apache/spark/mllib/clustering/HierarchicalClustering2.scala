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

package org.apache.spark.mllib.clustering

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, norm => breezeNorm}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.random.XORShiftRandom

import scala.collection.{Map, mutable}

class ClusterTree2(
  val center: Vector,
  val records: Long,
  val variances: Vector,
  var parent: Option[ClusterTree2],
  var children: List[ClusterTree2]
) extends Serializable {

  def this(center: Vector, rows: Long, variances: Vector) = this(center, rows, variances,
    None, List.empty[ClusterTree2])

  /**
   * Inserts sub nodes as its children
   *
   * @param children inserted sub nodes
   */
  def insert(children: List[ClusterTree2]) {
    this.children = this.children ++ children
    children.foreach(child => child.parent = Some(this))
  }

  /**
   * Inserts a sub node as its child
   *
   * @param child inserted sub node
   */
  def insert(child: ClusterTree2) {
    insert(List(child))
  }
}

class HierarchicalClustering2(
  private[mllib] var numClusters: Int,
  private[mllib] var clusterMap: Map[Int, ClusterTree2]
) {

  /**
   * Constructs with the default configuration
   */
  def this() = this(20, mutable.ListMap.empty[Int, ClusterTree2])

  def run(data: RDD[Vector]) {
    validate(data)

    val clusterdData = initializeData(data)

    // TODO: return the root node of a cluster tree
  }

  def validate(data: RDD[Vector]) {
  }

  private[clustering]
  def initializeData(data: RDD[Vector]): RDD[(Int, BV[Double])] = {
    data.map { v: Vector => (1, v.toBreeze)}.cache
  }

  private[clustering]
  def getCenterStats(data: RDD[(Int, BV[Double])]): Map[Int, ClusterTree2] = {

    // summarize data by each cluster
    val stats = data.mapPartitions { iter =>
      // calculate the accumulation of the all point in a partition and count the rows
      val map = mutable.Map.empty[Int, (BV[Double], Double, BV[Double])]
      iter.foreach { case (idx: Int, point: BV[Double]) =>
        // get a map value or else get a sparse vector
        val (sumBV, n, sumOfSquares) = map.get(idx)
            .getOrElse(BSV.zeros[Double](point.size), 0.0, BSV.zeros[Double](point.size))
        map(idx) = (sumBV + point, n + 1.0, sumOfSquares + (point :* point))
      }
      map.toIterator
    }.reduceByKey { case ((sv1, n1, sumOfSquares1), (sv2, n2, sumOfSquares2)) =>
      // sum the accumulation and the count in the all partition
      (sv1 + sv2, n1 + n2, sumOfSquares1 + sumOfSquares2)
    }

    // make clusters
    stats.collect().map { case (i, (sum, n, sumOfSquares)) =>
      val center = Vectors.fromBreeze(sum :/ n)
      val variances = n match {
        case n if n > 1 => Vectors.fromBreeze(sumOfSquares.:*(n) - (sum :* sum) :/ (n * (n - 1.0)))
        case _ => Vectors.sparse(sum.size, Array(), Array())
      }
      (i, new ClusterTree2(center, n.toLong, variances))
    }.toMap
  }

  /**
   * Takes the initial centers for bi-sect k-means
   */
  private[clustering]
  def takeInitCenters(center: BV[Double]): Array[BV[Double]] = {
    val random = new XORShiftRandom()
    Array(
      // TODO: add randomRange property
      // center.map(elm => elm - random.nextDouble() * elm * this.randomRange),
      // center.map(elm => elm + random.nextDouble() * elm * this.randomRange)
      center.map(elm => elm - random.nextDouble() * elm * 0.1),
      center.map(elm => elm + random.nextDouble() * elm * 0.1)
    )
  }

  private[clustering]
  def splitCenters(data: RDD[(Int, BV[Double])]): Map[Int, ClusterTree2] = {
    val sc = data.sparkContext
    // generate initial centers
    var clusters = getCenterStats(data)
    val nextCenters = clusters.map { case (i, c) => (i, takeInitCenters(c.center.toBreeze))}
    sc.broadcast(nextCenters)

    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)
    sc.broadcast(metric)

    val stats = data.mapPartitions { iter =>
      val map = mutable.Map.empty[Int, (BV[Double], Double, BV[Double])]
      iter.foreach { case (idx, point) =>
        // calculate next index number
        val closestIndex = HierarchicalClustering2.findClosestCenter(metric)(nextCenters(idx))(point)
        val nextIndex = 2 * idx + closestIndex + 1
        // get a map value or else get a sparse vector
        val (sumBV, n, sumOfSquares) = map.get(nextIndex)
            .getOrElse(BSV.zeros[Double](point.size), 0.0, BSV.zeros[Double](point.size))
        map(nextIndex) = (sumBV + point, n + 1.0, sumOfSquares + (point :* point))
      }
      map.toIterator

      // calculate the accumulation of the all point in a partition and count the rows
      //      val map = scala.collection.mutable.Map.empty[Int, (BV[Double], Long)]
      //      iter.foreach { case (idx, point) =>
      //        val centers = nextCenters(idx)
      //        val closestIndex = HierarchicalClustering2.findClosestCenter(metric)(centers)(point)
      //        val nextIndex = 2 * idx + closestIndex + 1
      //        val (sumBV, n) = map.get(nextIndex)
      //            .getOrElse((new BSV[Double](Array(), Array(), point.size), 0L))
      //        map(nextIndex) = (sumBV + point, n + 1L)
      //      }
      //      map.toIterator
    }.reduceByKey { case ((sv1, n1, sumOfSquares1), (sv2, n2, sumOfSquares2)) =>
      // sum the accumulation and the count in the all partition
      (sv1 + sv2, n1 + n2, sumOfSquares1 + sumOfSquares2)
    }

    // make clusters
    stats.collect().map { case (i, (sum, n, sumOfSquares)) =>
      val center = Vectors.fromBreeze(sum :/ n)
      val variances = n match {
        case n if n > 1 => Vectors.fromBreeze(sumOfSquares.:*(n) - (sum :* sum) :/ (n * (n - 1.0)))
        case _ => Vectors.sparse(sum.size, Array(), Array())
      }
      (i, new ClusterTree2(center, n.toLong, variances))
    }.toMap
  }

  def split(data: RDD[(Int, BV[Double])]): RDD[(Int, BV[Double])] = {
    // get next centers
    // optimize the next centers
    // assign the each data to the next centers
    data
  }
}

object HierarchicalClustering2 {

  private[mllib]
  def findClosestCenter(metric: Function2[BV[Double], BV[Double], Double])
        (centers: Array[BV[Double]])
        (point: BV[Double]): Int = {
    centers.zipWithIndex.map { case (center, idx) => (idx, metric(center, point))}.minBy(_._2)._1
  }
}
