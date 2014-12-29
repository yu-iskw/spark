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
import org.apache.spark.Logging
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

class HierarchicalClusteringModel2(val tree: ClusterTree2) extends Serializable with Logging {
}

class HierarchicalClustering2(
  private[mllib] var numClusters: Int,
  private[mllib] var clusterMap: Map[Int, ClusterTree2]
) extends Logging {

  /**
   * Constructs with the default configuration
   */
  def this() = this(20, mutable.ListMap.empty[Int, ClusterTree2])

  def setNumClusters(numClusters: Int): this.type = {
    this.numClusters = numClusters
    this
  }

  def run(input: RDD[Vector]): HierarchicalClusteringModel2 =  {
    validate(input)

    var data = initializeData(input).cache()

    // `clusters` is described as binary tree structure
    // `clusters(1)` means the root of the binary tree
    var clusters = getCenterStats(data)
    var numSplittedClusters = 0
    var noMoreSplit = false
    var step = 1
    while (numSplittedClusters < this.numClusters && noMoreSplit == false) {
      println(s"==== STEP:${step} is started")
      val splittedClusters = split(data, clusters)
      println(numSplittedClusters)
      println(splittedClusters.size)
      // enough to be clusterd if the number of splitted clusters is equal to 0
      if (splittedClusters.size > 0) {
        // update each index
        data = assign(data, splittedClusters)
        // merge the splitted clusters with the map as the cluster tree
        clusters = clusters ++ splittedClusters
        numSplittedClusters = data.map(_._1).distinct().count().toInt
        step += 1
      }
      else {
        noMoreSplit = true
      }
    }

    val root = clusters(1)
    new HierarchicalClusteringModel2(root)
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
        case _ => Vectors.zeros(sum.size)
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
  def split(
    data: RDD[(Int, BV[Double])],
    clusters: Map[Int, ClusterTree2]): Map[Int, ClusterTree2] = {
    val sc = data.sparkContext

    var newCenters: Map[Int, BV[Double]] = clusters.flatMap { case (idx, cluster) =>
      takeInitCenters(cluster.center.toBreeze).zipWithIndex
          .map { case (center, i) => (2 * idx + i, center)}
    }
    sc.broadcast(newCenters)

    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)
    sc.broadcast(metric)

    // TODO modify how to set iterations with a variable
    val vectorSize = newCenters(newCenters.keySet.min).size
    var stats = newCenters.keys.map { idx =>
      (idx, (BSV.zeros[Double](vectorSize).toVector, 0.0, BSV.zeros[Double](vectorSize).toVector))
    }.toMap
    // TODO stop if a relative error is very small
    for (i <- 1 to 20) {
      val eachStats = data.mapPartitions { iter =>
        val map = mutable.Map.empty[Int, (BV[Double], Double, BV[Double])]
        iter.foreach { case (idx, point) =>
          // calculate next index number
          val centers = Array(2 * idx, 2 * idx + 1).filter(newCenters.keySet.contains(_)).map(newCenters(_)).toArray
          if (centers.size >= 2) {
            val closestIndex = HierarchicalClustering2.findClosestCenter(metric)(centers)(point)
            val nextIndex = 2 * idx + closestIndex
            // get a map value or else get a sparse vector
            val (sumBV, n, sumOfSquares) = map.get(nextIndex)
                .getOrElse(BSV.zeros[Double](point.size), 0.0, BSV.zeros[Double](point.size))
            map(nextIndex) = (sumBV + point, n + 1.0, sumOfSquares + (point :* point))
          }
        }
        map.toIterator
      }.reduceByKey { case ((sv1, n1, sumOfSquares1), (sv2, n2, sumOfSquares2)) =>
        // sum the accumulation and the count in the all partition
        (sv1 + sv2, n1 + n2, sumOfSquares1 + sumOfSquares2)
      }.collect().toMap

      newCenters = eachStats.map { case (idx, (sum, n, sumOfSquares)) =>
        (idx, sum :/ n)
      }
      stats = eachStats
    }

    // make children clusters
    stats.filter { case (i, (sum, n, sumOfSquares)) => n > 0}
        .map { case (i, (sum, n, sumOfSquares)) =>
      val center = Vectors.fromBreeze(sum :/ n)
      val variances = Vectors.fromBreeze(sumOfSquares.:*(n) - (sum :* sum) :/ (n * (n - 1.0)))
      val child = new ClusterTree2(center, n.toLong, variances)
      // relate with parent cluster
      val parent = clusters((i / 2).toInt)
      parent.insert(child)
      (i, child)
    }.toMap
  }

  private[clustering]
  def assign(data: RDD[(Int, BV[Double])],
    clusters: Map[Int, ClusterTree2]): RDD[(Int, BV[Double])] = {
    val sc = data.sparkContext
    var centers = clusters.map { case (idx, cluster) => (idx, cluster.center)}
    sc.broadcast(centers)

    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)
    sc.broadcast(metric)

    data.map { case (idx, point) =>
      val indexes = Array(2 * idx, 2 * idx + 1).filter(centers.keySet.contains(_))
      indexes.size match {
        case s if s < 2 => (idx, point)
        case _ => {
          val nextCenters = indexes.map(centers(_)).map(_.toBreeze)
          val closestIndex = HierarchicalClustering2.findClosestCenter(metric)(nextCenters)(point)
          val nextIndex = 2 * idx + closestIndex
          (nextIndex, point)
        }
      }
    }
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
