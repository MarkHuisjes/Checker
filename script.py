#SECTION 1 - ARCGIS - create area and quantity variables
def geo():
    import arcpy
    arcpy.env.workspace = "D:/Projects/Thesis/Script/Data/Intermediate"

    # Prepare data in ArcGIS
    # Define input data
    inp = "D:/Projects/Thesis/Script/Data/Input/GEMDATAnp.shp"
    #Project to RD_New
    desc = arcpy.Describe(inp)
    sr = desc.spatialReference
    if sr.type == "Projected":
        fc = inp
    elif sr.type == "Geographic": 
        arcpy.Project_management (inp, "D:/Projects/Thesis/Script/Data/Intermediate/prj.shp", "D:/Projects/Thesis/Script/Data/Input/RDNew.prj")
        fc = "D:/Projects/Thesis/Script/Data/Intermediate/prj.shp"
    elif sr.type == "Unknown":
        arcpy.DefineProjection_management(inp, "D:/Projects/Thesis/Script/Data/Input/WGS84.prj")
        arcpy.Project_management (inp, "D:/Projects/Thesis/Script/Data/Intermediate/prj.shp", "D:/Projects/Thesis/Script/Data/Input/RDNew.prj")
        fc = "D:/Projects/Thesis/Script/Data/Intermediate/prj.shp"
    # Calculate area
    arcpy.AddGeometryAttributes_management(fc, "AREA", "KILOMETERS", "SQUARE_KILOMETERS")

    # Export attribute table
    # Set input output locations
    table = "D:/Projects/Thesis/Script/Data/Intermediate/prj.dbf"
    outfile = "D:/Projects/Thesis/Script/Data/Intermediate/prj.txt"
    # List fields and write field names
    fields = arcpy.ListFields(table)
    i = 1
    f = open(outfile,'w')
    for field in fields:
        #--write all field names to the output file
        if i < len(fields):
            f.write('%s,' % field.name)
            i += 1
        else:
            f.write('%s\n' % field.name)
    #Iterate through the records and write
    rows = arcpy.SearchCursor(table)
    for row in rows:
        i = 1
        for field in fields:
            if i < len(fields):
                f.write('%s,' % row.getValue(field.name))
                i += 1
            else:
                f.write('%s\n' % row.getValue(field.name))
    del rows
    f.close()
    #Change .txt to .csv
    import os
    Datatable = "D:/Projects/Thesis/Script/Data/Intermediate/prj.txt"
    base = os.path.splitext(Datatable)[0]
    os.rename(Datatable, base + ".csv")

geo()

#SECTION 2 - R - regress area and quantity variables
def stat():
    import subprocess
    
    subprocess.call (["C:/Program Files/R/R-3.4.0/bin/Rscript", "--vanilla", "D:/Projects/Thesis/Script/src/Rgr.r"])
        
        ##! "C:/Program Files/R/R-3.4.0/bin/Rscript

        #library("plyr")

        #df <- read.csv("D:/Projects/Thesis/Script/Data/Intermediate/prj.csv")
        ## throw away automatic columns
        #df$FID <- NULL
        #df$Shape <- NULL

        ##regress and determine residuals
            #reg.lm = lm(Bedrijfsve~POLY_AREA, data=df)
            #reg.res = resid(reg.lm)

        ##Calculate basic homoscedasticity values
            #MRabs=mean(abs(reg.res))
            #Tabs=mean(abs(df$Bedrijfsve))
            #avRes=mean(reg.res)
            #mdRes=median(reg.res)
        ##Calculate normalized average and median resisduals
            #NmAV=avRes/MRabs
            #NmMD=mdRes/MRabs
        ##Calculate normalized average and median resisduals per tritile
            #R=seq_along(reg.lm$residuals)
            #Nflt=order(R, decreasing=TRUE)[1:1]
            #Nint=trunc((Nflt)/3)
            #SPLT=split(reg.res, ceiling(seq_along(reg.res)/Nint))
            #avRes1=mean(SPLT$"1")
            #mdRes1=median(SPLT$"1")
            #avRes2=mean(SPLT$"2")
            #mdRes2=median(SPLT$"2")
            #avRes3=mean(SPLT$"3")
            #mdRes3=median(SPLT$"3")
            #NmAV1=avRes1/MRabs
            #NmMD1=mdRes1/MRabs
            #NmAV2=avRes2/MRabs
            #NmMD2=mdRes2/MRabs
            #NmAV3=avRes3/MRabs
            #NmMD3=mdRes3/MRabs
            
        ##Check linearity values
            #RsqdInv=1-(summary(reg.lm)$adj.r.squared)
            #NmInt=(coef(reg.lm)[1])/MRabs
            #NmSlp=(coef(reg.lm)[2])/Tabs
            
        ##Export results            
            #L=list("RsqdInv"=RsqdInv, "NmInt"=NmInt, "NmSlp"=NmSlp, "NmAV"=NmAV, "NmMD"=NmMD, "NmAV1"=NmAV1, "NmMD1"=NmMD1, "NmAV2"=NmAV2, "NmMD2"=NmMD2, "NmAV3"=NmAV3, "NmMD3"=NmMD3)
            #df=ldply(L, data.frame)
            #write.csv(df, "D:/Projects/Thesis/Script/Data/Intermediate/vals.csv")

stat()

#SECTION 3 - Type the geodata
def typing():
    
    #Clean up table, Check for tensity
    import pandas as pd
    f=pd.read_csv("D:/Projects/Thesis/Script/Data/Intermediate/vals.csv")
    keep_col = ['.id','X..i..']
    new_f = pd.DataFrame(f[keep_col])
    new_f.columns = ['Measure','Value']
    #Check calculated values against 
    new_f['Reference']=[0.3,0.2,0.95,0.1,50.1,0.1,0.1,0.1,0.1,0.1,0.1]
    new_f['Check']=('Value'>'Reference')
    new_f.to_csv("D:/Projects/Thesis/Script/Data/Output/TensityMeasures.csv", index=False)
    
    #Output tensity check
    if (sum(new_f.Check) > 0):
        print "Intensive"
    else:
        print "Extensive"
typing()