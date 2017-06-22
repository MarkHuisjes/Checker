#import glob
#import os.path

def prep(inp):
    import arcpy
    import os.path
    arcpy.env.workspace = "D:/Projects/Thesis/ScriptData/Input"
    
    fn = os.path.basename(inp)
    field_list = arcpy.ListFields(inp)
    throwFields = []
    for field in field_list:
            throwFields.append(field.name)
    #Remove required fields from throwFields
    if 'Shape.STLength()' in throwFields:
        throwFields.remove('Shape.STLength()')
    if 'Shape' in throwFields:
        throwFields.remove('Shape')
    if 'SHAPE.STLength()' in throwFields:
        throwFields.remove('SHAPE.STLength()')
    if 'Shape.STArea()' in throwFields:
        throwFields.remove('Shape.STArea()')
    if 'SHAPE.STArea()' in throwFields:
        throwFields.remove('SHAPE.STArea()')
    if 'SHAPE.AREA' in throwFields:
        throwFields.remove('SHAPE.AREA')
    if 'SHAPE.LEN' in throwFields:
        throwFields.remove('SHAPE.LEN')
    if 'SHAPE.LEN' in throwFields:
        throwFields.remove('SHAPE.LEN')
    if 'SHAPE' in throwFields:
        throwFields.remove('SHAPE')
    if 'Shape' in throwFields:
        throwFields.remove('Shape')
    if 'OBJECTID' in throwFields:
        throwFields.remove('OBJECTID')
    if 'FID' in throwFields:
        throwFields.remove('FID')
    #print throwFields
    for i in enumerate(throwFields):
        tmpfile = arcpy.CopyFeatures_management(inp, ("D:/Projects/Thesis/ScriptData/Input/"+str(i[1])+"_"+fn))
        keepFields = [str(i[1])]
        dropFields = [x for x in throwFields if x not in keepFields]
        arcpy.DeleteField_management(("D:/Projects/Thesis/ScriptData/Input/"+str(i[1])+"_"+fn), dropFields)
        with arcpy.da.UpdateCursor(("D:/Projects/Thesis/ScriptData/Input/"+str(i[1])+"_"+fn), str(i[1])) as cursor:
            for row in cursor:
                if row[0] == -99999999:
                    cursor.deleteRow()

def area(inp,cwd): #project, calculate area
    import arcpy
    import pandas as pd
    from simpledbf import Dbf5
    import os
    from shutil import copyfile
    import fnmatch
    import csv
    import re
    
    #Define paths
    arcpy.env.workspace = os.path.join(cwd, "Intermediate")
    rdpcs = os.path.join(cwd, "PRJ_Files/RDNew.prj")
    wgsgcs = os.path.join(cwd, "PRJ_Files/WGS84.prj")
    outp = os.path.join(cwd, "Intermediate/prj.shp")
    conv = os.path.join(cwd, "Intermediate/prj.dbf")
    csvloc = os.path.join(cwd, "Intermediate/prj.csv")
    rephtml1 = "C:/Users/Mark/AppData/Local/Temp/MoransI_Result.html"
    rephtml2 = "C:/Users/Mark/AppData/Local/Temp/GeneralG_Result.html"
    Itxt = os.path.join(cwd, "Intermediate/I.txt")
    Gtxt = os.path.join(cwd, "Intermediate/G.txt")
     
    #Project to RD_New
    desc = arcpy.Describe(inp)
    sr = desc.spatialReference
    if sr.type == "Projected":
        arcpy.CopyFeatures_management(inp, outp)
    elif sr.type == "Geographic": 
        arcpy.Project_management (inp, outp, rdpcs)
    elif sr.type == "Unknown":
        arcpy.DefineProjection_management(inp, wgsgcs)
        arcpy.Project_management (inp, outp, rdpcs)
    #Calculate area
    arcpy.AddGeometryAttributes_management(outp, "AREA_GEODESIC", "KILOMETERS", "SQUARE_KILOMETERS")
    #Save dbf as csv
    dbf = Dbf5(conv)
    qa = dbf.to_dataframe()
    valfield = str(qa.columns.values[0])
    qa.columns = ['Value', 'Area']
    qa.to_csv(csvloc)
    #Calculate spatial statistics and write them to file
    moransI = arcpy.SpatialAutocorrelation_stats(inp, valfield,
                        "GENERATE_REPORT", "INVERSE_DISTANCE", 
                        "EUCLIDEAN_DISTANCE", "NONE")
    copyfile(rephtml1, Itxt)
    G = arcpy.HighLowClustering_stats(inp, valfield, 
                        "GENERATE_REPORT", 
                        "INVERSE_DISTANCE",
                        "EUCLIDEAN_DISTANCE", "NONE")
    copyfile(rephtml2, Gtxt)
    with open (Itxt, 'rt') as in_file1:
        contents1 = str(in_file1.read())
        text1 = contents1.replace(" ","").replace("=","").replace(":","").replace("(","").replace(")","").replace("/","").replace("\\","").replace("<","").replace(">","").replace("'","").replace(";","").replace("#","").replace("{","").replace("}","").replace("\"","")
        lastline1 = text1[-1200:]
        M = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lastline1)
        N = [M[3], M[4]]
        spatstat1 = [float(i) for i in N]
        in_file1.close()
    with open (Gtxt, 'rt') as in_file1:
        contents1 = str(in_file1.read())
        text2 = contents1.replace(" ","").replace("=","").replace(":","").replace("(","").replace(")","").replace("/","").replace("\\","").replace("<","").replace(">","").replace("'","").replace(";","").replace("#","").replace("{","").replace("}","").replace("\"","")
        lastline2 = text2[-1200:]
        M = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lastline2)
        #print M
        N = [M[3], M[4]]
        spatstat2 = [float(i) for i in N]
        in_file1.close()
    global spatstat    
    spatstat = spatstat1+spatstat2
    os.remove(rephtml1)
    os.remove(rephtml2)
#End def area
        
def stat(cwd, inp): #Calculate statistics on spatial data
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression as lm
    from scipy.stats import levene as lv
    import random
    import csv
    import os
    import os.path
    import shutil
    #Define paths
    csvloc = os.path.join(cwd, "Intermediate/prj.csv")
    outploc= os.path.join(cwd, "Output/vals.csv")
    tensloc = os.path.join(cwd, "PRJ_Files/Tensity.csv")

    #Create normalized regressable parameters    
    qa = pd.read_csv(csvloc)
    aq = pd.concat([qa['Area'], qa['Value']], axis=1, keys=['Area', 'Value'])
    naq = (aq - aq.mean()) / (aq.max() - aq.min())
    npMatrix = np.matrix(naq)
    X, Y = npMatrix[:,0], npMatrix[:,1]
    #Regress normalized values
    rgrvl = lm().fit(X,Y)
    #Statistical measures on values
    slp = rgrvl.coef_[0]
    intc = rgrvl.intercept_
    R2 = rgrvl.score(X,Y)
    W, pval = lv(X, Y, center='median')
    resid = (rgrvl.predict(X))-Y
    MN_res = np.mean(resid) #Shouild be zero
    MD_res = np.median(resid, axis=0, overwrite_input=True)
    Rtab = naq.corr(method='pearson')
    R = Rtab.get_value('Area', 'Value', takeable=False)
    #Statistical measures on residuals
    rgrrs = lm().fit(X,resid)
    slprs = rgrrs.coef_[0]
    intcrs = rgrrs.intercept_
    R2rs = rgrrs.score(X,Y)
    residrs = (rgrrs.predict(X))-Y
    mnresrs = np.mean(residrs)
    mdresrs = np.median(residrs, axis=0, overwrite_input=True)
    #Collect statistics for export
    #statnames = ['slp', 'intc', 'R2', 'MN_res', 'MD_res', 'R', 'slprs', 'intcrs', 'R2rs', 'mnresrs', 'mdresrs', 'W', 'I', 'Zi', 'G', 'Zg']
    statlst = [slp, intc, R2, MN_res, MD_res, R, slprs, intcrs, R2rs, mnresrs, mdresrs, W, spatstat[0], spatstat[1], spatstat[2], spatstat[3]]
    statvalsnotens = [float(i) for i in statlst]
	
    #Collect tensity and statistics
    tens = pd.read_csv(tensloc)
    fnm = str(os.path.basename(inp))
    fn = fnm.replace("_","").replace(".shp","")
    mask = tens['A'] == fn
    lf = tens[mask]
    gf = lf.reset_index()
    del gf['index']
    tensity = gf.iloc[0]['B']
    statvals = [statvalsnotens, tensity]
    untransposed = pd.DataFrame(statvals)
    results = pd.DataFrame.transpose(untransposed)
    
    #Export stats to csv
    if os.path.isfile(outploc):
        with open(outploc, 'a') as f:
            (results).to_csv(f, header=False)
    else:
        results.to_csv(outploc)  
    #Delete data intermediate folder
    shutil.rmtree(arcpy.env.workspace)
    os.mkdir(arcpy.env.workspace)
#End def stat

#@profile
def classify():
    import clf
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np
    import os.path
    #Models
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.naive_bayes import GaussianNB
    #Peripheral
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix
    from collections import Counter
    import cProfile

    #Define paths
    modlloc = "D:/Projects/Thesis/NEW/Thesis/ScriptData/Output/Main/TrainTestNOzg.csv"
    #testloc = os.path.join(cwd, "Output/vals1.csv")
    #Model settings
    names = [#"Logistic Regression",
            "Nearest Neighbors", "Decision Tree", "Random Forest", "AdaBoost",
            "Neural Net", "Gaussian Process",
            "Linear SVM", "RBF SVM", "Naive Bayes"]
    classifiers = [
            #LogisticRegression(C=1e5),
            KNeighborsClassifier(3),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10,max_features=1),
            AdaBoostClassifier(),
            MLPClassifier(alpha=1),
            GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianNB()]

    #Import data, create train set
    data1 = np.genfromtxt(modlloc, delimiter=',')
    data2 = np.delete(data1,(0), axis=0)
    data3 = np.delete(data2,(0), axis=1)
    npMatrix = np.matrix(data3)
    X = npMatrix[:,0:14] #Stat values
    y = data3[:,15] #Tensity
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    #Naive model (majority vote)   
    bc = Counter(y).most_common(1)[0]
    naiveaccuracy = float(bc[1])/float(len(y))
    print('naiveaccuracy; '+str(naiveaccuracy))
    classes = list(set(y))
    
    #Define plotting the CM
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    #Iterate over classifiers
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        
        #scores = cross_val_score(clf,X,y,cv=5)#,scoring='f1_macro')

        #score = clf.score(X_test, y_test)
        #print('classifier: '+name+' score: '+str(score))
        #print("CV accuracy {}: {} (+/- {})".format(name,scores.mean(),scores.std()))

        #scoretr = clf.score(X_train, y_train)
        #print('classifier: '+name+' score no train: '+str(scoretr))
        y_pred = clf.fit(X_train, y_train).predict(X)
    
        # Compute confusion matrix
        cnf = confusion_matrix(y, y_pred,labels=classes)
        np.set_printoptions(precision=2)
        accuracy = (float(cnf[0][0])+float(cnf[1][1]))/(float(cnf[1][1])+float(cnf[0][0])+float(cnf[1][0])+float(cnf[0][1]))
        ptsaccuracyincrease = ((accuracy/naiveaccuracy)-1)*100
        print ('classifier: '+name+' naive accuracy: '+str(accuracy))
        print ('classifier: '+name+' percentage improvement: '+str(ptsaccuracyincrease))

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(cnf, classes=classes,title='Confusion matrix for '+name)
#End def classify

#Check input folder for shp data
#wd = "D:/Projects/Thesis/ScriptData/"
#inpdir = os.path.join(wd, "Create/*.shp")
#jobs = glob.glob(inpdir)
#if len(jobs) > 0:
#    #Iterate over qued files
#    for filename in jobs:
#        prep(inp=filename)
#wd = "D:/Projects/Thesis/ScriptData/"
#inpdir = os.path.join(wd, "Input/*.shp")
#jobs = glob.glob(inpdir)
#if len(jobs) > 0:
#Iterate over qued files
#    for filename in jobs:
#        spatstat = []
#        area(inp=filename, cwd=wd)#Projects file if necessary, calculates area
#        stat(inp=filename, cwd=wd)#Calculates statistics
classify()

#Tries to predict tensity from statistics
#print "done"
#else:
    #print('No jobs provided.')