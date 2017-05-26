#! "C:/Program Files/R/R-3.4.0/bin/Rscript

library("plyr")

df <- read.csv("D:/Projects/Thesis/Script/Data/Intermediate/prj.csv")
# throw away automatic columns
df$FID <- NULL
df$Shape <- NULL

#regress and determine residuals
reg.lm = lm(Bedrijfsve~POLY_AREA, data=df)
reg.res = resid(reg.lm)

#Check homoscedasticity values
MRabs=mean(abs(reg.res))
Tabs=mean(abs(df$Bedrijfsve))
avRes=mean(reg.res)
mdRes=median(reg.res)
#Calculate normalized average and median resisduals
NmAV=avRes/MRabs
NmMD=mdRes/MRabs
#Calculate normalized average and median resisduals per tritile
R=seq_along(reg.lm$residuals)
Nflt=order(R, decreasing=TRUE)[1:1]
Nint=trunc((Nflt)/3)
SPLT=split(reg.res, ceiling(seq_along(reg.res)/Nint))
avRes1=mean(SPLT$"1")
mdRes1=median(SPLT$"1")
avRes2=mean(SPLT$"2")
mdRes2=median(SPLT$"2")
avRes3=mean(SPLT$"3")
mdRes3=median(SPLT$"3")
NmAV1=avRes1/MRabs
NmMD1=mdRes1/MRabs
NmAV2=avRes2/MRabs
NmMD2=mdRes2/MRabs
NmAV3=avRes3/MRabs
NmMD3=mdRes3/MRabs

#Check linearity values
RsqdInv=1-(summary(reg.lm)$adj.r.squared)
NmInt=(coef(reg.lm)[1])/MRabs
NmSlp=(coef(reg.lm)[2])/Tabs

#Export results
L=list("RsqdInv"=RsqdInv, "NmInt"=NmInt, "NmSlp"=NmSlp, "NmAV"=NmAV, "NmMD"=NmMD, "NmAV1"=NmAV1, "NmMD1"=NmMD1, "NmAV2"=NmAV2, "NmMD2"=NmMD2, "NmAV3"=NmAV3, "NmMD3"=NmMD3)
df=ldply(L, data.frame)
write.csv(df, "D:/Projects/Thesis/Script/Data/Intermediate/vals.csv")


