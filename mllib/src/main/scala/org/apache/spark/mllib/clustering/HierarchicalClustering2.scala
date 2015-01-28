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
import org.apache.spark.{Logging, SparkException}

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

  def run(input: RDD[Vector]): HierarchicalClusteringModel2 = {
    validate(input)

    var data = initializeData(input).cache()

    // `clusters` is described as binary tree structure
    // `clusters(1)` means the root of the binary tree
    var clusters = getCenterStats(data)
    var numSplittedClusters = 0
    var deepestClusters = clusters
    var noMoreSplit = false
    var step = 1
    while (numSplittedClusters < this.numClusters && noMoreSplit == false) {
      log.info(s"==== STEP:${step} is started")

      // enough to be clusterd if the number of splitted clusters is equal to 0
      val splittedClusters = split(data, deepestClusters)
      if (splittedClusters == 0) {
        noMoreSplit = true
      }
      else {
        // update each index
        data = assign(data, splittedClusters)
        // merge the splitted clusters with the map as the cluster tree
        clusters = clusters ++ splittedClusters
        numSplittedClusters = data.map(_._1).distinct().count().toInt
        deepestClusters = splittedClusters
        step += 1
      }
    }

    val root = buildClusterTree(clusters, 1)
    if (root == None) {
      new SparkException("Failed to build a cluster tree from a Map type of clusters")
    }
    new HierarchicalClusteringModel2(root.get)
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

    // create clusters
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
      center.map(elm => elm - random.nextDouble() * elm * 0.0001),
      center.map(elm => elm + random.nextDouble() * elm * 0.0001)
    )
  }

  private[clustering]
  def takeInitCenters(data: RDD[(Int, BV[Double])]): Map[Int, BV[Double]] = {
    val sc = data.sparkContext
    val rand = new XORShiftRandom()
    sc.broadcast(rand)
    data.map { case (idx, point) =>
      val nextIndex = (rand.nextBoolean()) match {
        case true => 2 * idx
        case false => 2 * idx + 1
      }
      (nextIndex, (point, 1))
    }.reduceByKey { case ((point1, n1), (point2, n2)) => (point1 + point2, n1 + n2)}
        .map { case (idx, (point, n)) => (idx, point :/ n.toDouble)}
        .collect().toMap
  }

  private[clustering]
  def split(data: RDD[(Int, BV[Double])],
    clusters: Map[Int, ClusterTree2]): Map[Int, ClusterTree2] = {
    val sc = data.sparkContext

    // get keys of splittable clusters
    val splittableKeys = clusters.filter { case (idx, cluster) =>
      cluster.variances.toArray.sum > 0.0 && cluster.records >= 2
    }.keySet
    if (splittableKeys.size == 0) {
      log.info("There is no splittable clusters.")
      return Map.empty[Int, ClusterTree2]
    }

    // split input data
    var splittableData = data.filter { case (idx, point) => splittableKeys.contains(idx)}
    val idealIndexes = splittableKeys.flatMap(idx => Array(2 * idx, 2 * idx + 1).toIterator)
    var stats = __split(splittableData)

    // if there is clusters which is failed to be splitted,
    // retry to split only failed clusters again and again
    // TODO: max retries
    var numRetries = 1
    while (stats.size != splittableKeys.size * 2 && numRetries <= 20) {
      // get the indexes of clusters which is failed to be split
      val failedIndexes = idealIndexes.filterNot(stats.keySet.contains).map(idx => (idx / 2).toInt)
      log.info(s"# failed clusters: ${failedIndexes.size} at trying:${numRetries}")

      // split the failed clusters again
      sc.broadcast(failedIndexes)
      splittableData = data.filter { case (idx, point) => failedIndexes.contains(idx)}
      val missingStats = __split(splittableData)
      stats = stats ++ missingStats
      numRetries += 1
    }

    // make children clusters
    stats.filter { case (i, (sum, n, sumOfSquares)) => n > 0}
        .map { case (i, (sum, n, sumOfSquares)) =>
      val center = Vectors.fromBreeze(sum :/ n)
      val variances = Vectors.fromBreeze(sumOfSquares.:*(n) - (sum :* sum) :/ (n * (n - 1.0)))
      val child = new ClusterTree2(center, n.toLong, variances)
      (i, child)
    }.toMap
  }

  private[clustering]
  def buildClusterTree(treeMap: Map[Int, ClusterTree2], idx: Int): Option[ClusterTree2] = {
    if (!treeMap.contains(idx)) return None

    val parent = treeMap(idx)
    val indexes = Array(2 * idx, 2 * idx + 1)
    if (indexes.forall(i => treeMap.contains(i))) {
      val children = indexes.map(i => treeMap(i)).toList
      indexes.foreach(i => buildClusterTree(treeMap, i))
      parent.insert(children)
    }
    Some(parent)
  }

  private[clustering]
  def __split(data: RDD[(Int, BV[Double])]): Map[Int, (BV[Double], Double, BV[Double])] = {
    val sc = data.sparkContext
    var newCenters = takeInitCenters(data)
    sc.broadcast(newCenters)

    // TODO Supports distance metrics other Euclidean distance metric
    val metric = (bv1: BV[Double], bv2: BV[Double]) => breezeNorm(bv1 - bv2, 2.0)
    sc.broadcast(metric)

    // TODO modify how to set iterations with a variable
    val vectorSize = newCenters(newCenters.keySet.min).size
    var stats = newCenters.keys.map { idx =>
      (idx, (BSV.zeros[Double](vectorSize).toVector, 0.0, BSV.zeros[Double](vectorSize).toVector))
    }.toMap

    for (i <- 1 to 20) {
      // calculate summary of each cluster
      val eachStats = data.mapPartitions { iter =>
        val map = mutable.Map.empty[Int, (BV[Double], Double, BV[Double])]
        iter.foreach { case (idx, point) =>
          // calculate next index number
          val centers = Array(2 * idx, 2 * idx + 1).filter(newCenters.keySet.contains(_))
              .map(newCenters(_)).toArray
          if (centers.size >= 1) {
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

      // calculate the center of each cluster
      newCenters = eachStats.map { case (idx, (sum, n, sumOfSquares)) => (idx, sum :/ n)}
      // update summary of each cluster
      stats = eachStats
    }
    stats
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

    //    val keys = clusters.map(_._1).toArray.sorted
    //    println("== in assign method")
    //    println(s"min:${keys.min}, max:${keys.max}, size:${keys.size}")
    //    println(s"${keys.mkString(",")}")

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
