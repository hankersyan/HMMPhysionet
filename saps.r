load("setA_static_meanImpute_day1.rda")
load("setA_static_meanImpute_day2.rda")
print (head(y))
saps <- y$SAPSII
apache <-y$APACHEII
mpm <- y$MPMII
sofa <-y$SOFA
recid <- y$RecordID
# cat (recid)