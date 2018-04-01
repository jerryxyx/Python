setwd("~/Codes/Python/CourseProjects/FE542/Midterm/")
getwd()
df <- read.csv("CRSP_on_SP500_20y.csv")
head(df)
summary(df)
log_returns <- diff(log(df$totval),lag=1)
sarima_model <- arima(log_returns,order=c(1,0,1),seasonal=list(order=c(0,1,0),period=252))
sarima_model$aic
sarima_model <- arima(log_returns,order=c(1,0,1),seasonal=list(order=c(0,1,0),period=252))
criteria.matrix = matrix(0,nrow=4,ncol=4)
#for(i in 0:2){
#    for(j in 0:2){
#        sarima_model <- arima(log_returns,order=c(i,0,j),seasonal=list(order=c(0,1,0),period=252))
#        criteria.matrix[i+1,j+1] <- sarima_model$aic
#    }
#}
criteria.matrix
