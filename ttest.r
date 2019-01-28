MyData <- data.frame(read.csv(file="paperpercentage124-3.csv", header=TRUE, sep=","))
b1 = (MyData$hmmmse1)
a1= (MyData$baselinescore1)
res1 <- t.test(b1, a1, paired=TRUE, alternative= "less")
print (res1)
b2 = (MyData$hmmmse2)
a2= (MyData$baselinescore2)
res2 <- t.test(b2, a2, paired=TRUE, alternative= "less")
print (res2)