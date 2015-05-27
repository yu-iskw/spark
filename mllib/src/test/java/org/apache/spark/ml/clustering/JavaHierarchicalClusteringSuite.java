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

package org.apache.spark.ml.clustering;

import java.io.Serializable;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

public class JavaHierarchicalClusteringSuite implements Serializable {
  private transient JavaSparkContext jsc;
  private transient SQLContext jsql;

  @Before
  public void setUp() {
    jsc = new JavaSparkContext("local", "JavaHierarchicalClustering");
    jsql = new SQLContext(jsc);
  }

  @After
  public void tearDown() {
    jsc.stop();
    jsc = null;
  }

  @Test
  public void testFitAndPredict() {
    HierarchicalClustering algo = new HierarchicalClustering()
        .setFeaturesCol("point")
        .setNumClusters(5)
        .setSeed(1);

    DataFrame df = HierarchicalClusteringSuite.generateTestDataFrame(jsql, 5, 99);
    HierarchicalClusteringModel model = algo.fit(df);
    assertEquals(5, model.getCenters().length);
    assertEquals(5, model.getClusters().length);

    for (int i = 0; i < model.getCenters().length; i++) {
      Vector point = model.getCenters()[i];
      int idx = model.predict(point);
      assertEquals(i, idx);
    }
  }
}
