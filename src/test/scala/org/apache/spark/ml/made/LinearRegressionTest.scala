package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import org.apache.spark.ml
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.001
  lazy val data: DataFrame = LinearRegressionTest._data
  lazy val smallData: DataFrame = LinearRegressionTest._smallData
  lazy val vectors: Seq[Vector] = LinearRegressionTest._vectors
  lazy val coefficients: org.apache.spark.ml.linalg.DenseVector = LinearRegressionTest._coefficients
  lazy val intercept: Double = LinearRegressionTest._intercept

  println(data.printSchema())
  "Model" should "produce result" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      coefficients = coefficients,
      intercept = intercept
    ).setInputCol("features")
      .setOutputCol("target")

    val result: Array[Double] = model
      .transform(smallData.select("features"))
      .select("target")
      .collect()
      .map(_.getAs[Double](0))
    result.length should be(2)

    result(0) should be(
      vectors.head(0) * coefficients(0)
        + vectors.head(1) * coefficients(1)
        + vectors.head(2) * coefficients(2)
        + intercept
        +- delta
    )
    result(1) should be(
      vectors(1)(0) * coefficients(0)
        + vectors(1)(1) * coefficients(1)
        + vectors(1)(2) * coefficients(2)
        + intercept
        +- delta)
  }

  "Estimator" should "calculates coefficients" in {
    val estimator: LinearRegression = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("target")

    val model: LinearRegressionModel = estimator.fit(data)
    println(model.coefficients)
    model.coefficients.size should be(3)
    model.coefficients(0) should be (coefficients(0) +- delta)
  }

}

object LinearRegressionTest extends WithSpark {
  lazy val _vectors = Seq(
    Vectors.dense(0.1, 0.2, 0.3),
    Vectors.dense(0.4, 0.3, 0.2)
  )

  import spark.implicits._

  lazy val _coefficients: ml.linalg.DenseVector = Vectors.dense(1.5, 0.3, -0.7).toDense
  lazy val _intercept: Double = .0

  lazy val _X: breeze.linalg.DenseMatrix[Double] = breeze.linalg.DenseMatrix.rand(100000, 3)
  lazy val _y: breeze.linalg.DenseVector[Double] = _X * _coefficients.asBreeze
  lazy val _df: DenseMatrix[Double] = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)



  lazy val _smallData: DataFrame = {
    import sqlc.implicits._
      _vectors.map(x =>
        (x, (x dot _coefficients) + _intercept)
      ).toDF("features", "target")
  }

  lazy val _data: DataFrame = _df(*, ::).iterator
    .map(x => (Vectors.dense(x(0), x(1), x(2)), x(3)))
    .toSeq.toDF("features", "target")

}