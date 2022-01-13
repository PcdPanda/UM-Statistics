X = read.csv("ETH-USD.csv",header=TRUE)
ETHwk <- rev(X$Close)
plot(ETHwk)
ETHwk_lreturn <- diff(log(ETHwk)) # generating log returns (weekly)

plot(ETHwk_lreturn,type='l')
XX = read.csv("ETH-USD.csv",header=TRUE)
ETHdl <- rev(XX$Close)

plot(ETHdl,type='l')
ETHdl_lreturn <- diff(log(ETHdl)) # generating difference in log(daily closing price)

plot(ETHdl_lreturn)
par(mfrow=c(2,2)) # setting up for a 2 x 2 arrangement of subplots
plot(ETHwk,xlab='week',ylab='closing price',type='l')
plot(ETHwk_lreturn,xlab='week',ylab='wkly log return',type='l')
plot(ETHdl,xlab='day',ylab='closing price',type='l')
plot(ETHdl_lreturn,xlab='day',ylab='daily log return',type='l')
dim(XX)
summary(XX)