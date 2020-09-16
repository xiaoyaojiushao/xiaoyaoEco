# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:55:47 2020

@author: 逍遥九少
"""

# 本文件用于基于余弦相似度的纵向和横向文本相似度的计算


import numpy as np
import pandas as pd
import jieba
import os
from sklearn.metrics.pairwise import cosine_similarity

PROVINCE_LIST = ['北京','天津','河北','山西','内蒙古','辽宁','吉林','黑龙江','上海','江苏','浙江','安徽','福建','江西','山东','河南','湖北','湖南','广东','广西','海南','重庆','四川','贵州','云南','西藏','陕西','甘肃','青海','宁夏','新疆']


def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5*cos
    return sim

def simliarity_horizon(field = '意见', wordNum = 30,is_showResult = False, is_saveResult = False):
    # 这里只计算省级的，省事一点
    # 先把所有的文件合并（注意是同类型的），然后得出一个baseWordList
    global PROVINCE_LIST
    
    provfiles = os.listdir('./省级')
    #提取出有该领域文件的省份
    targProvs = []
    targProvFiles = []
    for provfile in provfiles:
        if field in provfile:
            tempSplit = provfile.split()
            targProvs.append(tempSplit[0])
            targProvFiles.append(provfile)
    # 提取完毕，现在将这些文件合并然后分词返回baselist
    # 这里的baseFreqList 没什么用
    baseWordList, baseFreqList = cut_for_word_and_list(targProvFiles, wordNum, model = '横向')
    del baseFreqList
    
    # 再对各个省份分别进行分词，每个省形成一个自己的FreqList，返回一个字典
    cmpFreqDict = countCmpList(baseWordList,field,
                 PROVINCE_LIST,model = '横向',
                 targCity=targProvs,targCityFiles=targProvFiles,
                 prov = '河南', )
    # 最后，针对有Freq的省份之间计算相似度，没有的就不管了
    # 要定义一个返回显示都矩阵的新函数
    simMat = clacSimMat(cmpFreqDict,targProvs)
    simMat = pd.DataFrame(simMat,columns = targProvs, index = targProvs)
     
    if is_showResult:
        simMat.head()
    
    if is_saveResult:
        filename = './横向相似度结果/' + field + ' - ' + '各省之间的横向相似度矩阵' + '.csv'
        simMat.to_csv(filename, encoding='utf-8_sig')
    
    
    return simMat

def clacSimMat(cmpFreqDict,targProvs):
    # 要返回一个相似度矩阵，
    # 我记得我之前做过类似的！
    # 我想一下
    val = [i for i in cmpFreqDict.values()]
    simMat = cosine_similarity(val)
    return simMat


def similiarity_vertical(model = '中央-省', prov = '河南', field = '意见',
                         wordNum = 30, is_showResult = False, is_saveResult = False):
    #本 函数计算的是纵向相似度
    global PROVINCE_LIST
    
    if model == '中央-省':
        # 对中央分词
        filename = './中央/中央-' + field + '.txt'
        baseWordList, baseFreqList = cut_for_word_and_list(filename, wordNum, model)
        
        # 用已有的词列表来计算省级去情况
        # 这里就有个问题，是全部统一计算呢还是分开计算呢
        # 算了，统一计算即可
        cmpFreqDict = countCmpList(baseWordList, field, 
                                   PROVINCE_LIST,model)
        
        # 接下来就是计算各个省份于中央的相似度啦
        # 计算的结果也应该是一个dict
        simDict = clacSim(baseFreqList, cmpFreqDict,PROVINCE_LIST)
        
        # 将dict保存为df
        simDF = pd.DataFrame.from_dict(simDict, orient='index',columns=['与该领域中央文件的相似度'])
        if is_showResult:
            simDF.head()
            
        if is_saveResult:
            # 先把dict保存成df，然后再保存
            filename = './纵向相似度-中央-省/' + field + ' - 各省与中央该领域文件的纵向相似度.csv'
            simDF.to_csv(filename, encoding='utf-8_sig')
            
             
    elif model == '省-市':
        # 由于市级的这个，没有一个完整的名单
        # 所以在搞之前需要先统计一个名单
        
        # 先判断这个省份有没有市级，没有直接跳过 
        prov_have_city_list = os.listdir('./市级')
        if not( prov in prov_have_city_list):
            print(prov+'省没有市级的文件')
            gg = []
            return gg
        # 这个省里面的市级没有该领域的文件，仍然跳过
        city_files = os.listdir('./市级/' + prov + '/')
        flag = False
        for cityfile in city_files:
            if field in cityfile:
                flag = True
        if not(flag): 
            print(prov+'省没有市级该领域的文件')
            gg = []
            return gg
        
        # 搞个名单
        # 这个名单是一个省份，里面有相关文件的地级市，的市名
        # 现在，这个文件夹是存在的，让我们打开他
        cityFiles = os.listdir('./市级/' + prov + '/')
        targCityFiles = []
        targCity = []
        for file in cityFiles:
            if field in file:
                targCityFiles.append(file)
                temp = file.split('-')
                city = temp[0]
                targCity.append(city)

        # 搞完名单之后，先统计省级该领域的，作为base
        # 首先这个省份得有省级的文件啊
        filename = './省级/' + prov + '-' + field + '.txt'
        if not(os.path.exists(filename)):
            print(prov+'省没有省级的base的文件')
            gg = []
            return gg
        baseWordList, baseFreqList = cut_for_word_and_list(filename, wordNum, model)
        
        # 然后统计市级该领域的，作为cmp
        # 在那个函数里面加参数，把city们传进去
        # 难度就在这里了，不过可以克服
        cmpFreqDict = countCmpList(baseWordList, field, 
                                   PROVINCE_LIST,model,
                                   targCity,targCityFiles,prov)
        # 计算相似度
        # 这个难度已经不大了
        simDict = clacSim(baseFreqList, cmpFreqDict,targCity)
        simDF = pd.DataFrame.from_dict(simDict, orient='index',columns=['与该领域省级文件的相似度'])
        if is_showResult:
            simDF.head()
            
        if is_saveResult:
            # 先把dict保存成df，然后再保存
            filename = './纵向相似度-省-市/' + prov + ' - ' + field + ' - 各市与省级该领域文件的纵向相似度.csv'
            simDF.to_csv(filename, encoding='utf-8_sig')

    return simDict



def cut_and_return_items(filename):
    f = open(filename,encoding='utf-8')
    tempstr = f.read()
    f.close()
    words = jieba.lcut(tempstr)
    counts = {}
    for word in words:
        if len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word,0) + 1
    items = list(counts.items())
    items.sort(key = lambda x:x[1], reverse = True)
    return items

        
def cut_for_word_and_list(filename, wordNum, model):
    if model == '中央-省' or model == '省-市':
        items = cut_and_return_items(filename)        
        #将前N个数组转化成list
        baseWordList = []
        baseFreqList = []
        for i in list(range(wordNum)):
            baseWordList.append(items[i][0])
            baseFreqList.append(items[i][1])
    elif model == '横向':
        # 先把这些文件给合并了
        longtext = ''
        for file in filename:
            fullname = './省级/' + file
            f = open(fullname,encoding='utf-8')
            tempstr = f.read()
            f.close()
            longtext = longtext + tempstr
        # 现在开始分词
        words = jieba.lcut(longtext)
        counts = {}
        for word in words:
            if len(word) == 1:
                continue
            else:
                counts[word] = counts.get(word,0) + 1
        items = list(counts.items())
        items.sort(key = lambda x:x[1], reverse = True)
        baseWordList = []
        baseFreqList = []
        for i in list(range(wordNum)):
            baseWordList.append(items[i][0])
            baseFreqList.append(items[i][1])
            
    return baseWordList, baseFreqList
        
def countCmpList(baseWordList,field,PROVINCE_LIST,model,targCity=[],targCityFiles=[],prov = '河南'):
    # 目标为cmpFreqDict
    if model == '中央-省':
        # 先看看哪些省份有，有就做，没有就归为空列表
        provFiles = os.listdir('./省级')
        # 制作一个list，每个省对应一个数值list，最后用zip缝起来
        rdyList = []
        for prov in PROVINCE_LIST:
            provList = []
            targetName = prov + '-' + field + '.txt'
            if targetName in provFiles:
                fullname = './省级/' + targetName
                items = cut_and_return_items(fullname)
                
                # 从items里面找对应的词语的频率
                for word in baseWordList:
                    for item in items:
                        if word == item[0]:
                            provList.append(item[1])
                
                # 如果说有的词里面没有，则赋值0
                if len(provList) < 30:
                    needZero = 30-len(provList)
                    for i in range(needZero):
                        provList.append(0)
            else:
                provList = []
            # 把结果添加到一个大的list里    
            rdyList.append(provList)
        
        # 合并成dict形式
        cmpFreqDict = dict(zip(PROVINCE_LIST,rdyList))
        
        return cmpFreqDict
    elif model == '省-市':
        # 我们返回的是一个cityDict，是每一个city对应一个list
        # 最后是将cityList和rdyList合并成一个Dict
        
        rdyList = []
        # 由于我们已经有了文件名，所以直接遍历就完事了
        for cityFile in targCityFiles:
            cityList = []
            fullname = './市级/' + prov + '/' + cityFile 
            items = cut_and_return_items(fullname)
            # 从items里面找对应的词语的频率
            for word in baseWordList:
                for item in items:
                    if word == item[0]:
                        cityList.append(item[1])
                
            # 如果说有的词里面没有，则赋值0
            if len(cityList) < 30:
                needZero = 30-len(cityList)
                for i in range(needZero):
                    cityList.append(0)
            rdyList.append(cityList)
        
        cmpFreqDict = dict(zip(targCity,rdyList))
        return cmpFreqDict
    elif model == '横向':
        # 每一个省份对应一个list
        rdyList = []
        # 由于我们已经有了文件名，所以直接遍历就完事了
        for cityFile in targCityFiles:
            cityList = []
            fullname = './省级/' + cityFile 
            items = cut_and_return_items(fullname)
            # 从items里面找对应的词语的频率
            for word in baseWordList:
                for item in items:
                    if word == item[0]:
                        cityList.append(item[1])
                
            # 如果说有的词里面没有，则赋值0
            if len(cityList) < 30:
                needZero = 30-len(cityList)
                for i in range(needZero):
                    cityList.append(0)
            rdyList.append(cityList)
        
        cmpFreqDict = dict(zip(targCity,rdyList))
        return cmpFreqDict
            
def clacSim(baseFreqList, cmpFreqDict,IterList):
    # 用于计算相似度，返回一个dict
    simList = []
    for prov in IterList:
        tempCmpList = cmpFreqDict[prov]
        if len(tempCmpList) == 0:
            simList.append(0)
        else:
            tempSim = cos_sim(baseFreqList, tempCmpList)
            simList.append(tempSim)
    simDict = dict(zip(IterList, simList))
    return simDict

fieldlist = ['意见' ,'教育', '医疗', '科技' ,'交通', '公共']

for province in PROVINCE_LIST:
    for onefield in fieldlist:
        mydict = similiarity_vertical(model = '省-市', prov = province, field = onefield,
                         wordNum = 30, is_showResult = False, is_saveResult = True)

for onefield in fieldlist:
    mydict = similiarity_vertical(model = '中央-省', prov = province, field = onefield,
                         wordNum = 30, is_showResult = False, is_saveResult = True)
    mydict2 = simliarity_horizon(field = onefield, wordNum = 30,is_saveResult = True)


#mydict2 = simliarity_horizon(field = '意见', wordNum = 30,is_saveResult = True)