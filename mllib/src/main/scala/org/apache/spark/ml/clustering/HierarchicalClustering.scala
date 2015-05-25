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
package org.apache.spark.ml.clustering

import org.apache.spark.annotation.AlphaComponent
import org.apache.spark.ml.param._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

/**
 * :: AlphaComponent ::
 *
 * TODO write a description for the hierarchical clustering parameters.
 */
@AlphaComponent
private[clustering]
trait HierarchicalClusteringParams extends Params with HasMaxIter with HasFeaturesCol {

  /**
   * Param for the number of clusters you want.
   * @group param
   */
  val numClusters = new IntParam(this, "numClusters", "number of clusters you want")

  def getNumClusters: Int = get(numClusters)

  /**
   * Param for the maximum number of retries
   * @group param
   */
  val maxRetries = new IntParam(this, "maxRetries", "maximum number of retries")

  def getMaxRetries: Int = get(maxRetries)

  /**
   * Param for a random seed
   * @group param
   */
  val seed = new IntParam(this, "seed", "random seed")

  def getSeed: Int = get(seed)

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @param paramMap extra params
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    val map = this.paramMap ++ paramMap
    assert(schema(map(featuresCol)).dataType == ArrayType)
    schema
  }
}

/**
 * :: AlphaComponent ::
 *
 * TODO write a description for the hierarchical clustering.
 */
@AlphaComponent
class HierarchicalClustering
    extends Estimator[HierarchicalClusteringModel] with HierarchicalClusteringParams {

  setMaxIter(20)
  setMaxRetries(5)
  setSeed(1)

  /** @group setParam */
  def setNumClusters(value: Int): this.type = set(numClusters, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setMaxRetries(value: Int): this.type = set(maxRetries, value)

  /** @group setParam */
  def setSeed(value: Int): this.type = set(seed, value)

  def setInputCol(value: String): this.type = set(featuresCol, value)

  /**
   * Fits a single model to the input data with provided parameter map.
   *
   * @param dataset input dataset
   * @param paramMap Parameter map.
   *                 These values override any specified in this Estimator's embedded ParamMap.
   * @return fitted model
   */
  override def fit(dataset: DataFrame, paramMap: ParamMap): HierarchicalClusteringModel = {
    val map = this.paramMap ++ paramMap
    val oldData = extractVector(dataset, paramMap)
    val algo = new mllib.clustering.HierarchicalClustering()
        .setNumClusters(map(numClusters))
        .setMaxIterations(map(maxIter))
        .setMaxRetries(map(maxRetries))
        .setSeed(map(seed))
    val parentModel = algo.run(oldData)
    val model = new HierarchicalClusteringModel(this, map, parentModel)
    model
  }

  def extractVector(dataset: DataFrame, paramMap: ParamMap): RDD[Vector] = {
    val map = this.paramMap ++ paramMap
    dataset.select(col(map(featuresCol))).map { case Row(point: Vector) => point }
  }

  /**
   * :: DeveloperApi ::
   *
   * Derives the output schema from the input schema and parameters.
   * The schema describes the columns and types of the data.
   *
   * @param schema  Input schema to this stage
   * @param paramMap  Parameters passed to this stage
   * @return  Output schema from this stage
   */
  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    validateAndTransformSchema(schema, paramMap)
  }
}

/**
 * :: AlphaComponent ::
 *
 * Model produced by [[HierarchicalClustering]].
 */
@AlphaComponent
class HierarchicalClusteringModel private[ml](
    override val parent: HierarchicalClustering,
    override val fittingParamMap: ParamMap,
    val parentModel: mllib.clustering.HierarchicalClusteringModel)
    extends Model[HierarchicalClusteringModel] with HierarchicalClusteringParams {

  def predict(point: Vector): Int = {
    parentModel.predict(point)
  }

  override def transform(dataset: DataFrame, paramMap: ParamMap): DataFrame = dataset

  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = schema

  def toLinkageMatrix(): Array[(Int, Int, Double, Int)] = parentModel.toLinkageMatrix()

  def toJavaLinkageMatrix(): java.util.ArrayList[java.util.ArrayList[java.lang.Double]] =
    parentModel.toJavaLinkageMatrix()

  def toAdjacencyList(): Array[((Int, Int, Double))] = parentModel.toAdjacencyList()

  def toJavaAdjacencyList(): java.util.ArrayList[java.util.ArrayList[java.lang.Double]] =
      parentModel.toJavaAdjacencyList()
}
