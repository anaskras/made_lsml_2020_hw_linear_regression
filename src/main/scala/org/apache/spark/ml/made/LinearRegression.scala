package org.apache.spark.ml.made

import breeze.linalg
import breeze.linalg.{InjectNumericOps, norm}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}


trait LinearRegressionParams extends HasInputCol with HasOutputCol { //
  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

class LinearRegression(override val uid: String, val fitIntercept: Boolean)
  extends Estimator[LinearRegressionModel]
    with LinearRegressionParams
    with DefaultParamsWritable {

  def this(fitIntercept: Boolean) = this(Identifiable.randomUID("linearRegression"), fitIntercept)

  def this() = this(Identifiable.randomUID("linearRegression"), false)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    def grad(weights: linalg.DenseVector[Double], intercept: Double, row: Row) = {
      val features = row.getAs[Vector](0).asBreeze
      val answer = row.getDouble(1)
      val error = features.dot(weights) + intercept - answer
      (error * features, error)
    }

    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val target = linalg.DenseVector(dataset
      .select($(outputCol))
      .collect()
      .map(_.getAs[Double](0))
    )

    val n_samples: Double = target.size

    val n_features = dataset
      .select($(inputCol))
      .first()(0)
      .asInstanceOf[DenseVector]
      .size

    var intercept: Double = 0
    val weights: linalg.DenseVector[Double] = linalg.DenseVector.ones[Double](n_features)
    val learning_rate: Double = 0.1

    val rdd = dataset
      .select($(inputCol), $(outputCol))
      .rdd

    val num_of_iterations = 2000
    var i = 0
    var converged = false
//    for (i <- 0 to num_of_iterations) {
    while (!converged && i < num_of_iterations) {
      val (deltaWeights, deltaIntercept) = rdd
        .map(rw => grad(weights, intercept, rw))
        .reduce((a, b) => (a._1 + b._1, a._2 + b._2))

      weights -= (learning_rate / n_samples) * deltaWeights
      intercept -= (learning_rate / n_samples) * deltaIntercept

      if (norm((learning_rate / n_samples) * deltaWeights) < 0.000001) {
        converged = true
        println(i)
      }
    i += 1
    }

    copyValues(
      new LinearRegressionModel(
        Vectors.fromBreeze(weights).toDense,
        intercept))
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}


object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String,
                                          val coefficients: DenseVector,
                                          val intercept: Double
                                         )
  extends Model[LinearRegressionModel] with LinearRegressionParams{

  private[made] def this(uid: String) = this(uid, Vectors.zeros(0).toDense, 0)

  private[made] def this(coefficients: DenseVector, intercept: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), coefficients, intercept)

  override def copy(extra: ParamMap): LinearRegressionModel = {
      copyValues(new LinearRegressionModel(coefficients, intercept), extra)
    }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.select($(inputCol)).sqlContext.udf.register(uid + "_transform", {
      (x: Vector) => (x dot coefficients) + intercept
    })
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}


