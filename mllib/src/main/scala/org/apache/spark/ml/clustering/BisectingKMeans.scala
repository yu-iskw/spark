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

import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.param.{IntParam, ParamMap, Params}
import org.apache.spark.ml.util.{Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib.clustering.{BisectingKMeans => MLlibBisectingKMeans, BisectingKMeansModel => MLlibBisectingKMeansModel}
import org.apache.spark.mllib.linalg.{Vector, VectorUDT}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{IntegerType, StructType}
import org.apache.spark.sql.{DataFrame, Row}


/**
 * Common params for BisectingKMeans and BisectingKMeansModel
 */
private[clustering] trait BisectingKMeansParams extends Params
  with HasMaxIter with HasFeaturesCol with HasSeed with HasPredictionCol {

  /**
   * Set the number of clusters to create (k). Must be > 1. Default: 2.
   * @group param
   */
  @Since("1.6.0")
  final val k = new IntParam(this, "k", "number of clusters to create", (x: Int) => x > 1)

  /** @group getParam */
  @Since("1.6.0")
  def getK: Int = $(k)

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new VectorUDT)
    SchemaUtils.appendColumn(schema, $(predictionCol), IntegerType)
  }
}

/**
 * :: Experimental ::
 * Model fitted by KMeans.
 *
 * @param parentModel a model trained by spark.mllib.clustering.KMeans.
 */
@Since("1.6.0")
@Experimental
class BisectingKMeansModel private[ml] (
    @Since("1.6.0") override val uid: String,
    private val parentModel: MLlibBisectingKMeansModel
  ) extends Model[BisectingKMeansModel] with KMeansParams {

  @Since("1.6.0")
  override def copy(extra: ParamMap): BisectingKMeansModel = {
    val copied = new BisectingKMeansModel(uid, parentModel)
    copyValues(copied, extra)
  }

  @Since("1.6.0")
  override def transform(dataset: DataFrame): DataFrame = {
    val predictUDF = udf((vector: Vector) => predict(vector))
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  @Since("1.6.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  private[clustering] def predict(features: Vector): Int = parentModel.predict(features)

  @Since("1.6.0")
  def getCenters: Array[Vector] = parentModel.getCenters

  /**
   * Return the K-means cost (sum of squared distances of points to their nearest center) for this
   * model on the given data.
   */
  // TODO: Replace the temp fix when we have proper evaluators defined for clustering.
  @Since("1.6.0")
  def computeCost(dataset: DataFrame): Double = {
    SchemaUtils.checkColumnType(dataset.schema, $(featuresCol), new VectorUDT)
    val data = dataset.select(col($(featuresCol))).map { case Row(point: Vector) => point }
    parentModel.computeCost(data)
  }
}

/**
 * :: Experimental ::
 */
@Since("1.6.0")
@Experimental
class BisectingKMeans @Since("1.6.0") (
    @Since("1.6.0") override val uid: String)
  extends Estimator[BisectingKMeansModel] with BisectingKMeansParams {

  setDefault(
    k -> 2,
    maxIter -> 20)

  @Since("1.6.0")
  override def copy(extra: ParamMap): BisectingKMeans = defaultCopy(extra)

  @Since("1.6.0")
  def this() = this(Identifiable.randomUID("kmeans"))

  /** @group setParam */
  @Since("1.6.0")
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  @Since("1.6.0")
  def setK(value: Int): this.type = set(k, value)

  /** @group setParam */
  @Since("1.6.0")
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  @Since("1.6.0")
  def setSeed(value: Long): this.type = set(seed, value)

  @Since("1.6.0")
  override def fit(dataset: DataFrame): BisectingKMeansModel = {
    val rdd = dataset.select(col($(featuresCol))).map { case Row(point: Vector) => point }

    val algo = new MLlibBisectingKMeans()
      .setK($(k))
      .setMaxIterations($(maxIter))
      .setSeed($(seed))
    val parentModel = algo.run(rdd)
    val model = new BisectingKMeansModel(uid, parentModel)
    copyValues(model)
  }

  @Since("1.6.0")
  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
}

