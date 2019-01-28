library("ggplot2")
library ("ggforce")

cases = c("ord8","ordchg8","ordnonchg8","overlap8","overlapchg8","overlapnonchg8")
weightedcases = c("ordweighted8","ordchgweighted8","ordnonchgweighted8","overlapweighted8","overlapchgweighted8","overlapnonchgweighted8")

filteredcases = c("ordfiltered8","ordchgfiltered8","ordnonchgfiltered8","overlapfiltered8","overlapchgfiltered8","overlapnonchgfiltered8")

for (case in cases)
{
csvname = paste(c(case,".csv"),collapse = "")
# print(csvname)
MyData <- data.frame(read.csv(file=csvname, header=FALSE, sep=","))
params <- data.frame(read.csv(file="params.csv", header=FALSE, sep=","))
Resolution <- params$V1
Numstates  <- params$V2
icutype <- params$V3

# print (params)
# print(Resolution)
# print (Numstates)
startstates <- (MyData$V1)
endstates <- MyData$V2
values <- (MyData$V3)
numbs <- (MyData$V4)

preciseaxis = c(1:Numstates)



# circle generator 

circles <- data.frame(
  Start_State = startstates + 1,
  End_State =  endstates + 1,
  MeanLenghtOfStayX500 = values  ,
  numbpatsX25 = numbs  
)
realtitle = paste(c("Mean LOS for start-end state pairs resolution and #states and icutype", Resolution , Numstates,icutype,case ), collapse = " ")
print (realtitle)
fig = ggplot() + geom_circle(aes(x0=Start_State, y0=End_State, r=numbpatsX25, fill=MeanLenghtOfStayX500), data=circles)  + coord_fixed() + scale_x_discrete(limits = preciseaxis)+ scale_y_discrete(limits = preciseaxis) + scale_colour_gradient(limits=0:1) + scale_fill_gradient(limits=0:1)
print(fig + ggtitle(realtitle))
figtitle = paste(c("avglosstartendpairresnumstate", Resolution , Numstates ,icutype,case,".png"), collapse = "")
ggsave(figtitle)


scale_x_discrete(limits = preciseaxis)
scale_y_discrete(limits = preciseaxis)
scale_colour_gradient(limits=0:1) 
scale_fill_gradient(limits=0:1)

}


for (case in filteredcases)
{
csvname = paste(c(case,".csv"),collapse = "")
# print(csvname)
MyData <- data.frame(read.csv(file=csvname, header=FALSE, sep=","))
params <- data.frame(read.csv(file="params.csv", header=FALSE, sep=","))
Resolution <- params$V1
Numstates  <- params$V2
icutype <- params$V3


# print (params)
# print(Resolution)
# print (Numstates)
startstates <- (MyData$V1)
endstates <- MyData$V2
values <- (MyData$V3)
numbs <- (MyData$V4)

preciseaxis = c(1:Numstates)



# circle generator 

circles <- data.frame(
  Start_State = startstates + 1,
  End_State =  endstates + 1,
  MeanLenghtOfStayX500 = values  ,
  numbpatsX25 = numbs  

)
realtitle = paste(c("Mean LOS for start-end state pairs resolution and #states", Resolution , Numstates,icutype,case ), collapse = " ")
print (realtitle)
fig = ggplot() + geom_circle(aes(x0=Start_State, y0=End_State, r=numbpatsX25, fill=MeanLenghtOfStayX500), data=circles)  + coord_fixed() + scale_x_discrete(limits = preciseaxis)+ scale_y_discrete(limits = preciseaxis) + scale_colour_gradient(limits=0:1) + scale_fill_gradient(limits=0:1)
print(fig + ggtitle(realtitle))
figtitle = paste(c("avglosstartendpairresnumstate", Resolution , Numstates ,icutype,case,".png"), collapse = "")
ggsave(figtitle)


scale_x_discrete(limits = preciseaxis)
scale_y_discrete(limits = preciseaxis)
scale_colour_gradient(limits=0:1) 
scale_fill_gradient(limits=0:1)

}



for (case in weightedcases)
{
csvname = paste(c(case,".csv"),collapse = "")
# print(csvname)
MyData <- data.frame(read.csv(file=csvname, header=FALSE, sep=","))
params <- data.frame(read.csv(file="params.csv", header=FALSE, sep=","))
Resolution <- params$V1
Numstates  <- params$V2
icutype <- params$V3


# print (params)
# print(Resolution)
# print (Numstates)
startstates <- (MyData$V1)
endstates <- MyData$V2
values <- (MyData$V3)
numbs <- (MyData$V4)

preciseaxis = c(1:Numstates)



# circle generator 

circles <- data.frame(
  Start_State = startstates + 1,
  End_State =  endstates + 1,
  MeanLenghtOfStayX500 = values,
  numbpatsX25 = numbs  

)
realtitle = paste(c("Mean LOS for start-end state pairs resolution and #states", Resolution , Numstates,icutype,case ), collapse = " ")
print (realtitle)
fig = ggplot() + geom_circle(aes(x0=Start_State, y0=End_State, r=numbpatsX25, fill=MeanLenghtOfStayX500), data=circles)  + coord_fixed() + scale_x_discrete(limits = preciseaxis)+ scale_y_discrete(limits = preciseaxis)+ scale_colour_gradient(limits=0:1) + scale_fill_gradient(limits=0:1)
print(fig + ggtitle(realtitle))
figtitle = paste(c("avglosstartendpairresnumstate", Resolution , Numstates ,icutype,case,".png"), collapse = "")
ggsave(figtitle)


scale_x_discrete(limits = preciseaxis)
scale_y_discrete(limits = preciseaxis)
scale_colour_gradient(limits=0:1) 
scale_fill_gradient(limits=0:1)

}

