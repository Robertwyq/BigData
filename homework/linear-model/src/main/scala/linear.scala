import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object linear {
  def main(args: Array[String]) {
    val inputPath = args(0)
    val iterations = args(1).toInt

    val spark = SparkSession
      .builder
      .appName("linear")
      .getOrCreate()
      
    // Load and parse the data
    val data = spark.read.textFile(inputPath).rdd
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }.cache()

    // Building the model
    val model = LinearRegressionWithSGD.train(parsedData, iterations)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = parsedData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println("training Mean Squared Error = " + MSE)

    spark.stop()
  }
}