#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#' @include generics.R column.R
NULL

#' @title S4 expression functions for DataFrame column(s)
#' @description These are expression functions on DataFrame columns

functions1 <- c(
  "abs", "acos", "approxCountDistinct", "ascii", "asin", "atan",
  "avg", "base64", "bin", "bitwiseNOT", "cbrt", "ceil", "cos", "cosh", "count",
  "crc32", "dayofmonth", "dayofyear", "exp", "explode", "expm1", "factorial",
  "first", "floor", "hex", "hour", "initcap", "isNaN", "last", "last_day",
  "length", "log", "log10", "log1p", "log2", "lower", "ltrim", "max", "md5",
  "mean", "min", "minute", "month", "negate", "quarter", "reverse",
  "rint", "round", "rtrim", "second", "sha1", "signum", "sin", "sinh", "size",
  "soundex", "sqrt", "sum", "sumDistinct", "tan", "tanh", "toDegrees",
  "toRadians", "to_date", "trim", "unbase64", "unhex", "upper", "weekofyear",
  "year")
functions2 <- c(
  "atan2", "datediff", "hypot", "levenshtein", "months_between", "nanvl", "pmod")

createFunction1 <- function(name) {
  setMethod(name,
            signature(x = "Column"),
            function(x) {
              jc <- callJStatic("org.apache.spark.sql.functions", name, x@jc)
              column(jc)
            })
}

createFunction2 <- function(name) {
  setMethod(name,
            signature(y = "Column"),
            function(y, x) {
              if (class(x) == "Column") {
                x <- x@jc
              }
              jc <- callJStatic("org.apache.spark.sql.functions", name, y@jc, x)
              column(jc)
            })
}

createFunctions <- function() {
  for (name in functions1) {
    createFunction1(name)
  }
  for (name in functions2) {
    createFunction2(name)
  }
}

createFunctions()

#' @rdname functions
#' @return Creates a Column class of literal value.
#' @export
lit <- function(x) {
  jc <- callJStatic("org.apache.spark.sql.functions", "lit", ifelse(class(x) == "Column", x@jc, x))
  column(jc)
}

#' Approx Count Distinct
#'
#' @rdname functions
#' @return the approximate number of distinct items in a group.
setMethod("approxCountDistinct",
          signature(x = "Column"),
          function(x, rsd = 0.95) {
            jc <- callJStatic("org.apache.spark.sql.functions", "approxCountDistinct", x@jc, rsd)
            column(jc)
          })

#' Count Distinct
#'
#' @rdname functions
#' @return the number of distinct items in a group.
setMethod("countDistinct",
          signature(x = "Column"),
          function(x, ...) {
            jcol <- lapply(list(...), function (x) {
              x@jc
            })
            jc <- callJStatic("org.apache.spark.sql.functions", "countDistinct", x@jc,
                              listToSeq(jcol))
            column(jc)
          })

#' @rdname functions
#' @return Concatenates multiple input string columns together into a single string column.
setMethod("concat",
          signature(x = "Column"),
          function(x, ...) {
            jcols <- lapply(list(x, ...), function(x) { x@jc })
            jc <- callJStatic("org.apache.spark.sql.functions", "concat", listToSeq(jcols))
            column(jc)
          })

#' @rdname functions
#' @return Returns the greatest value of the list of column names, skipping null values.
#'         This function takes at least 2 parameters. It will return null if all parameters are null.
setMethod("greatest",
          signature(x = "Column"),
          function(x, ...) {
            stopifnot(length(list(...)) > 0)
            jcols <- lapply(list(x, ...), function(x) { x@jc })
            jc <- callJStatic("org.apache.spark.sql.functions", "greatest", listToSeq(jcols))
            column(jc)
          })

#' @rdname functions
#' @return Returns the least value of the list of column names, skipping null values.
#'         This function takes at least 2 parameters. It will return null iff all parameters are null.
setMethod("least",
          signature(x = "Column"),
          function(x, ...) {
            stopifnot(length(list(...)) > 0)
            jcols <- lapply(list(x, ...), function(x) { x@jc })
            jc <- callJStatic("org.apache.spark.sql.functions", "least", listToSeq(jcols))
            column(jc)
          })

#' @rdname functions
#' @aliases ceil
setMethod("ceiling",
          signature(x = "Column"),
          function(x) {
            ceil(x)
          })

#' @rdname functions
#' @aliases signum
setMethod("sign", signature(x = "Column"),
          function(x) {
            signum(x)
          })

#' @rdname functions
#' @aliases countDistinct
setMethod("n_distinct", signature(x = "Column"),
          function(x, ...) {
            countDistinct(x, ...)
          })

#' @rdname functions
#' @aliases count
setMethod("n", signature(x = "Column"),
          function(x) {
            count(x)
          })

#' @rdname functions
setMethod("date_format", signature(y = "Column", x = "character"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "date_format", y@jc, x)
            column(jc)
          })

#' @rdname functions
setMethod("from_utc_timestamp", signature(y = "Column", x = "character"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "from_utc_timestamp", y@jc, x)
            column(jc)
          })

#' @rdname functions
setMethod("instr", signature(y = "Column", x = "character"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "instr", y@jc, x)
            column(jc)
          })

#' @rdname functions
setMethod("next_day", signature(y = "Column", x = "character"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "next_day", y@jc, x)
            column(jc)
          })

#' @rdname functions
setMethod("to_utc_timestamp", signature(y = "Column", x = "character"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "to_utc_timestamp", y@jc, x)
            column(jc)
          })

#' @rdname functions
setMethod("add_months", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "add_months", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("date_add", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "date_add", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("date_sub", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "date_sub", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("format_number", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "format_number", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
#' @param y column to compute SHA-2 on.
#' @param x one of 224, 256, 384, or 512.
setMethod("sha2", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "sha2", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("shiftLeft", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "shiftLeft", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("shiftRight", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "shiftRight", y@jc, as.integer(x))
            column(jc)
          })

#' @rdname functions
setMethod("shiftRightUnsigned", signature(y = "Column", x = "numeric"),
          function(y, x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "shiftRightUnsigned", y@jc, as.integer(x))
            column(jc)
          })

#' complicated def array_contains(column: Column, value: Any): Column
#' rdname functions
setMethod("array_contains", signature(x = "Column"),
          function(x, y) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "array_contains", x@jc, y)
            column(jc)
          })

#' complicated def concat_ws(sep: String, exprs: Column*): Column
#' rdname functions
setMethod("concat_ws", signature(x = "Column"),
          function(x, y) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "concat_ws", x@jc, y)
            column(jc)
          })

#' complicated def conv(num: Column, fromBase: Int, toBase: Int): Column
#' rdname functions
setMethod("conv", signature(x = "Column", y = "numeric", z = "numeric"),
          function(x, y, z) {
            y <- as.integer(y)
            z <- as.integer(z)
            jc <- callJStatic("org.apache.spark.sql.functions", "conv", x@jc, y, z)
            column(jc)
          })

#' complicated def expr(expr: String): Column
#' rdname functions
setMethod("expr", signature(x = "character"),
          function(x) {
            jc <- callJStatic("org.apache.spark.sql.functions", "expr", x)
            column(jc)
          })

#' complicated def format_string(format: String, arguments: Column*): Column
#' rdname functions
setMethod("format_string", signature(x = "character", y = "Column"),
          function(x, y, ...) {
            jcols <- listToSeq(lapply(list(y, ...), function(x) { x@jc }))
            jc <- callJStatic("org.apache.spark.sql.functions", "format_string", x, jcols)
            column(jc)
          })

#' complicated def from_unixtime(ut: Column): Column
#' complicated def from_unixtime(ut: Column, f: String): Column
#' rdname functions
setMethod("from_unixtime", signature(x = "Column"),
          function(x, format = "yyyy-MM-dd HH:mm:ss") {
            jc <- callJStatic("org.apache.spark.sql.functions", "from_unixtime", x@jc, format)
            column(jc)
          })

#' complicated def locate(substr: String, str: Column): Column
#' complicated def locate(substr: String, str: Column, pos: Int): Column
#' rdname functions
setMethod("locate", signature(x = "character", y = "Column"),
          function(x, y, pos = 0) {
            jc <- callJStatic("org.apache.spark.sql.functions", "locate", x, y@jc, as.integer(pos))
            column(jc)
          })

#' complicated def lpad(str: Column, len: Int, pad: String): Column
#' rdname functions
setMethod("lpad", signature(x = "Column", len = "numeric", pad = "character"),
          function(x, len, pad) {
            jc <- callJStatic("org.apache.spark.sql.functions", "lpad", x@jc, as.integer(len), pad)
            column(jc)
          })

#' complicated def rand(seed: Long): Column
#' complicated def rand(): Column
#' rdname functions
setMethod("rand", signature(seed = "missing"),
          function(seed) {
            jc <- callJStatic("org.apache.spark.sql.functions", "rand")
            column(jc)
          })
setMethod("rand", signature(seed = "numeric"),
          function(seed) {
            jc <- callJStatic("org.apache.spark.sql.functions", "rand", as.integer(seed))
            column(jc)
          })

#' complicated def randn(): Column
#' complicated def randn(seed: Long): Column
#' rdname functions
setMethod("randn", signature(seed = "missing"),
          function(seed) {
            jc <- callJStatic("org.apache.spark.sql.functions", "randn")
            column(jc)
          })
setMethod("randn", signature(seed = "numeric"),
          function(seed) {
            jc <- callJStatic("org.apache.spark.sql.functions", "randn", as.integer(seed))
            column(jc)
          })

#' complicated def regexp_extract(e: Column, exp: String, groupIdx: Int): Column
#' rdname functions
setMethod("regexp_extract", signature(x = "Column", pattern = "character", idx = "numeric"),
          function(x, pattern, idx) {
            jc <- callJStatic("org.apache.spark.sql.functions", "regexp_extract", x@jc, pattern, as.integer(idx))
            column(jc)
          })

#' complicated def regexp_replace(e: Column, pattern: String, replacement: String): Column
#' rdname functions
setMethod("regexp_replace", signature(x = "Column", pattern = "character", replacement = "character"),
          function(x, pattern, replacement) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "regexp_replace", x@jc, pattern, replacement)
            column(jc)
          })

#' complicated def rpad(str: Column, len: Int, pad: String): Column
#' rdname functions
setMethod("rpad", signature(x = "Column", len = "numeric", pad = "character"),
          function(x, len, pad) {
            jc <- callJStatic("org.apache.spark.sql.functions", "rpad", x@jc, as.integer(len), pad)
            column(jc)
          })

#' complicated def substring_index(str: Column, delim: String, count: Int): Column
#' rdname functions
setMethod("substring_index", signature(x = "Column", delim = "character", count = "numeric"),
          function(x, delim, count) {
            jc <- callJStatic("org.apache.spark.sql.functions", "substring_index", x@jc, delim, as.integer(count))
            column(jc)
          })

#' complicated def translate(src: Column, matchingString: String, replaceString: String): Column
#' rdname functions
setMethod("translate", signature(x = "Column", matchingString = "character", replaceString = "character"),
          function(x, matchingString, replaceString) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "translate", x@jc, matchingString, replaceString)
            column(jc)
          })

#' complicated def unix_timestamp(s: Column, p: String): Column
#' complicated def unix_timestamp(): Column
#' complicated def unix_timestamp(s: Column): Column
#' rdname functions
setMethod("unix_timestamp", signature(x = "Column"),
          function(x, y) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "unix_timestamp", x@jc, y)
            column(jc)
          })

#' complicated def when(condition: Column, value: Any): Column
#' rdname functions
setMethod("when", signature(x = "Column"),
          function(x, y) {
            # TODO
            jc <- callJStatic("org.apache.spark.sql.functions", "when", x@jc, y)
            column(jc)
          })

