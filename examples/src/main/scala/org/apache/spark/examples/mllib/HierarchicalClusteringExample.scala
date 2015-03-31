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

package org.apache.spark.examples.mllib

import org.apache.spark.mllib.clustering.HierarchicalClustering
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

object HierarchicalClusteringExample {

  def main(args: Array[String]) {
    val parser = getOptionParser(args)
    parser.parse(args, new Params).map { params =>
      run(params)
    }.getOrElse(System.exit(1))
  }

  def run(params: Params) {
    // Creates a SparkContext
    val conf = new SparkConf().setAppName(s"HierarchicalClusteringExample with ${params}")
    val sc = new SparkContext(conf)

    // Loads an input data from a CSV or TSV file
    val input = sc.textFile(params.input)
    val data: RDD[Vector] = input.map(line => parseVector(line)).filter(_.isDefined).map(_.get)

    // Trains a hierarchical clustering model
    val numClusters = params.numClusters
    val maxIterations = params.maxIterations
    val maxRetries = params.maxRetries
    val seed = params.seed
    val model = HierarchicalClustering.train(data, numClusters, maxIterations, maxRetries, seed)

    // Shows the result
    println("Centers:")
    model.getClusters().foreach { case cluster =>
        println(s"${cluster.center}")
    }
    sc.stop()
  }

  def parseVector(line: String): Option[Vector] = {
    try {
      Some(Vectors.dense(line.split( """(,|\t)""").map(_.toDouble)))
    } catch {
      case e: NumberFormatException => None
    }
  }

  protected case class Params(
      input: String = null,
      numClusters: Int = Int.MaxValue,
      maxIterations: Int = 20,
      maxRetries: Int = 10,
      seed: Int = 1) extends AbstractParams[Params]

  /** Parses a command line options */
  def getOptionParser(args: Array[String]): OptionParser[Params] = {
    val defaultParams = new Params()
    val parser = new OptionParser[Params]("HierarchicalClusteringExample") {
      head("HierarchicalClusteringExample: an example app for Hierarchical Clustering.")

      arg[String]("<input>")
          .required()
          .text("input a CSV or TSV file")
          .action((x, c) => c.copy(input = x))
      arg[Int]("<numClusters>")
          .required()
          .text("number of clusters you want to divide")
          .action((x, c) => c.copy(numClusters = x))

      opt[Int]("maxIterations")
          .optional()
          .text(s"maximal number of iterations, default: ${defaultParams.maxIterations}")
          .action((x, c) => c.copy(maxIterations = x))
      opt[Int]("maxRetries")
          .optional()
          .text(s"maximal number of retries, default: ${defaultParams.maxRetries}")
          .action((x, c) => c.copy(maxRetries = x))
      opt[Int]("seed")
          .optional()
          .text(s"random seed, default: ${defaultParams.seed}")
          .action((x, c) => c.copy(seed = x))

      note(
        """
          |For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class org.apache.spark.examples.mllib.HierarchicalClusteringExample \
          |  examples/target/scala-*/spark-examples-*.jar \
          |  --maxIterations 20 --maxRetries 10 --seed 1 \
          |  data/mllib/sample_hierarchical_data.csv 3
        """.stripMargin)
    }
    parser
  }
}
