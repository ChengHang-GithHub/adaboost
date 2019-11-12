from numpy import *
import matplotlib.pyplot as plt
#===================================定义加载数据的函数==============================================
def loadDataFeature(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')#删除每一行中开头和结尾处的水平制表符
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)#得到一个样本数据
        labelMat.append(float(curLine[-1]))#得到一个样本的标签
    return dataMat,labelMat
def loadtestDataFeature(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')#删除每一行中开头和结尾处的水平制表符
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)#得到一个样本数据
        labelMat.append(float(curLine[-1]))#得到一个样本的标签
    return dataMat,labelMat
#定义单层决策树的分类函数

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):#输入参数为数据矩阵、特征、阈值、标志
    retArray = ones((shape(dataMatrix)[0], 1))#将一个列数为1，行数与原始数据矩阵相同的矩阵初始化为1。
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0   #将某一个特征的所有取值与阈值作比较，和阈值在同一边的分到类别-1；
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray
#===============================定义函数来表示单个弱分类器的训练过程====================================
def buildStump(dataArr, classLabels, D):#输入为数据矩阵、类别标签、权重向量（m行一列）
    dataMatrix = mat(dataArr)#将数据数组转化为矩阵形式
    labelMat = mat(classLabels).T  #转置运算
    m, n = shape(dataMatrix)
    numSteps = 10.0#步数
    bestStump = {}#建立一个空字典
    bestClasEst = mat(zeros((m, 1)))#建立一个m行一列的矩阵，并初始化为0
    minError = inf  # 将最小误差设置为无穷大
    for i in range(n):  # 遍历每一个特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps#步长
        for j in range(-1, int(numSteps) + 1):  #对于每一个特征，根据步长遍历可能的阈值
            threshVal = (rangeMin + float(j) * stepSize)
            for inequal in ['lt', 'gt']:  # 特征确定，阈值确定时遍历每一种标志
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)  # 调用单层决策树分类函数
                errArr = mat(ones((m, 1)))  #误差矩阵初始化为1
                errArr[predictedVals == labelMat] = 0#更新误差矩阵，预测对的更新为0
                weightedError = D.T * errArr  # 矩阵乘法得到加权错误率（一个数）
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst    #返回的是一个存储这个弱分类器参数的字典、最小误差、弱分类器的预测结果

#============================================定义函数用来训练（多个分类器）================================================
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  # 初始化权重矩阵（样本分布为均匀分布）
    aggClassEst = mat(zeros((m, 1)))#将分类结果矩阵（m*1)初始化为1
    for i in range(numIt):  #遍历每一次迭代，一次迭代可以生成一个弱分类器
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 调用单层决策树函数
        print ("D:",D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 计算α，得到每个分类器的权值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  # 用一个矩阵来存储最优单层决策树的参数
       # print ("classEst: ",classEst.T)
        #====================更新权重系数 D=================
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # Calc New D for next iteration
        D = D / D.sum()
        #===================================================
        aggClassEst += alpha * classEst#更新根据分类器取值更新预测输出结果
       # print ("aggClassEst: ",aggClassEst.T)
        #=======================计算错误率===============
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #===================================================
        print("total error: ", errorRate)
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst    #返回的是每次迭代弱分类器参数、分类器输出结果

#=====================================定义函数用来测试========================================
def adaClassify(datToClass, classifierArr,Lables):#输入为测试数据和训练好的分类器参数矩阵
    dataMatrix = mat(datToClass)  # 将输入数据转化为矩阵形式
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))#初始化分类结果矩阵为0
    for i in range(len(classifierArr)):#遍历每一个分类器
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])  # 调用分类函数
        aggClassEst += classifierArr[i]['alpha'] * classEst
    aggErrors = multiply(sign(aggClassEst) != mat(Lables).T, ones((m, 1)))
    errorRate = aggErrors.sum() / m
       # print(aggClassEst)
    return sign(aggClassEst),errorRate#返回最后一个分类器的分类结果（效果最好的分类器）

data_arr,class_lables=loadDataFeature("horseColicTraining2.txt")
weak_ClassArr, agg_ClassEst=adaBoostTrainDS(data_arr, class_lables, numIt=60)#得到训练之后各级弱分类器的参数，和最终的训练结果
print( weak_ClassArr)  #打印分类器的参数
data_arr,class_lables=loadtestDataFeature("horseColicTest2.txt")
a,err=adaClassify(data_arr, weak_ClassArr,class_lables)#调用函数用测试集进行测试
print("results: ",a)
print("errorrate:",err)



