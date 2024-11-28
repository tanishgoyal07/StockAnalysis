from datetime import datetime
from pyspark.sql.functions import udf
from pyspark.sql.types import DateType, FloatType, IntegerType
from pyspark.sql.window import Window
from pyspark.sql import SparkSession

import pyspark.sql.functions as func

def num_parser(value):
    if isinstance(value, str):
        return float(value.strip("$"))
    elif isinstance(value, int) or isinstance(value, float):
        return value
    else:
        return None

date_parser = udf(lambda date: datetime.strptime(date,"%m/%d/%Y"), DateType())
parser_number = udf(num_parser, FloatType())
parse_int = udf(lambda value: int(value), IntegerType())


spark = SparkSession.builder \
    .appName("Stock Market Analysis") \
    .master("local[*]") \
    .getOrCreate()

stocks = spark.read.csv("StockData", header=True)
stocks = stocks.withColumn("ParsedDate", date_parser(stocks.Date))

stocks = (stocks.withColumn("Open", parser_number(stocks.Open))
                .withColumn("Close", parser_number(stocks["Close/Last"]))
                .withColumn("Low", parser_number(stocks.Low))
                .withColumn("High", parser_number(stocks.High)))

stocks = stocks.withColumn("Volume", parse_int(stocks.Volume))

cleaned_stocks = stocks.select(["Ticker", "ParsedDate", "Volume", "Open", "Low", "High", "Close"])
cleaned_stocks.describe(["Volume", "Open", "Low", "High", "Close"]).show()

cleaned_stocks.groupBy("Ticker").max("Open").withColumnRenamed("max(Open)", "MaxStockPrice")
cleaned_stocks.show()

cleaned_stocks.groupBy("Ticker").agg(
    func.max("Open").alias("MaxStockPrice"),
    func.sum("Volume").alias("TotalVolume")
)
cleaned_stocks.show()

cleaned_stocks = (cleaned_stocks.withColumn("Year", func.year(cleaned_stocks.ParsedDate))
                                .withColumn("Month", func.month(cleaned_stocks.ParsedDate))
                                .withColumn("Day", func.dayofmonth(cleaned_stocks.ParsedDate))
                                .withColumn("Week", func.weekofyear(cleaned_stocks.ParsedDate))
                 )

yearly = cleaned_stocks.groupBy(['Ticker', 'Year']).agg(func.max("Open").alias("YearlHigh"), func.min("Open").alias("YearlLow"))
monthly = cleaned_stocks.groupBy(['Ticker', 'Year', 'Month']).agg(func.max("Open").alias("MonthHigh"), func.min("Open").alias("MonthLow"))
weekly = cleaned_stocks.groupBy(['Ticker', 'Year', 'Week']).agg(func.max("Open").alias("WeekHigh"), func.min("Open").alias("WeekLow"))

stock_df1 = cleaned_stocks.join(yearly, 
                    (cleaned_stocks.Ticker==yearly.Ticker) & (cleaned_stocks.Year == yearly.Year),
                    'inner'
                   ).drop(yearly.Year, yearly.Ticker)

cond = [(stock_df1.Ticker==weekly.Ticker) & (stock_df1.Year == weekly.Year) & (stock_df1.Week == weekly.Week)]
stock_df2 = stock_df1.join(weekly, cond, 'inner').drop(weekly.Year, stock_df1.Ticker, weekly.Week)

stock_df3 = stock_df2.join(monthly, ['Ticker', 'Year', 'Month'])

snapshot = cleaned_stocks.select(['Ticker', 'ParsedDate', 'Open'])
lag1Day = Window.partitionBy("Ticker").orderBy("ParsedDate")
snapshot.withColumn("PreviousOpen", func.lag("Open", 1).over(lag1Day)).show()

movingAverage = Window.partitionBy("Ticker").orderBy("ParsedDate").rowsBetween(-50, 0)
ma50 = (snapshot.withColumn("MA50", func.avg("Open").over(movingAverage))
                .withColumn("MA50", func.round("MA50", 2))
                .drop(snapshot.Open))

movingAverage = Window.partitionBy("Ticker").orderBy("ParsedDate").rowsBetween(-15, 0)
ma15 = (snapshot.withColumn("MA15", func.avg("Open").over(movingAverage))
                .withColumn("MA15", func.round("MA15", 2))
                .drop(snapshot.Open))

movingAverage = Window.partitionBy("Ticker").orderBy("ParsedDate").rowsBetween(-100, 0)
ma100 = (snapshot.withColumn("MA100", func.avg("Open").over(movingAverage))
                .withColumn("MA100", func.round("MA100", 2))
                .drop(snapshot.Open))

stock_df4 = stock_df3.join(ma50, ['Ticker', 'ParsedDate'])
stock_df5 = stock_df4.join(ma15, ['Ticker', 'ParsedDate'])
stock_df6 = stock_df5.join(ma100, ['Ticker', 'ParsedDate'])

(stock_df6.write.option("header",True)
             .partitionBy("Ticker")
             .mode("overwrite")
             .parquet("StockAnalysis_Parquet"))
             
(stock_df6.write.option("header",True)
             .partitionBy("Ticker")
             .mode("overwrite")
             .csv("StockAnalysis_CSV"))