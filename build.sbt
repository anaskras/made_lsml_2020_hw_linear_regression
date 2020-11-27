name := "hw7"

version := "0.1"

scalaVersion := "2.12.12"

val sparkVersion = "3.0.1"
val breezeVersion = "1.1"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion withSources(),
  "org.apache.spark" %% "spark-mllib" % sparkVersion withSources(),
  // Last stable release
  "org.scalanlp" %% "breeze" % breezeVersion withSources(),

  // Native libraries are not included by default. add this if you want them
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % breezeVersion withSources(),

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % breezeVersion  withSources()
)

libraryDependencies += ("org.scalatest" %% "scalatest" % "3.2.2" % "test" withSources())