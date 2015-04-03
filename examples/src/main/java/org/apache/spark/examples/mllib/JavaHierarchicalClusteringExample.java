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

package org.apache.spark.examples.mllib;

import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.HierarchicalClustering;
import org.apache.spark.mllib.clustering.HierarchicalClusteringModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

/**
 * Example using MLlib Hierarchical Clustering from Java.
 *
 * <pre>{@code
 * }</pre>
 */
public final class JavaHierarchicalClusteringExample {

  private static class ParsePoint implements Function<String, Vector> {
    private static final Pattern SPACE = Pattern.compile(",");

    @Override
    public Vector call(String line) {
      String[] tok = SPACE.split(line);
      double[] point = new double[tok.length];
      for (int i = 0; i < tok.length; ++i) {
        point[i] = Double.parseDouble(tok[i]);
      }
      return Vectors.dense(point);
    }
  }

  public static void main(String[] args) {
    if (args.length != 2) {
      System.err.println("Usage: JavaHierarchicalClusteringExample <csv_file> <num_clusters>");
      System.err.print("Example: ./bin/spark-submit ");
      System.err.println(" --class " + JavaHierarchicalClusteringExample.class.getName() + " \\");
      System.err.println("    examples/target/scala-*/spark-examples-*.jar \\");
      System.err.println("    ./data/mllib/sample_hierarchical_data.csv 3");
      System.exit(1);
    }
    String file = args[0];
    int k = Integer.parseInt(args[1]);

    SparkConf sparkConf = new SparkConf().setAppName("JavaHierarchicalClusteringExample");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    JavaRDD<String> lines = sc.textFile(file);
    JavaRDD<Vector> points = lines.map(new ParsePoint());

    HierarchicalClusteringModel model = HierarchicalClustering.train(points.rdd(), k);
    Vector[] centers = model.getCenters();
    System.out.println("Centers:");
    for (int i = 0; i < centers.length; ++i) {
      System.out.println(centers[i]);
    }

    sc.stop();
  }
}
