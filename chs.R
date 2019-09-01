##加载包
library("chinese.misc")
library("data.table")
library("stringr")
library("reshape")
library("jiebaR")
library("text2vec")

setwd("F:/a series of documents/陈华珊老师课件/关于论文数据")
##批量读取
files <- dir_or_file("F:/a series of documents/陈华珊老师课件/关于论文数据/wb", special='txt$')
nas <- files 
mylist <- lapply(files,function(i) scancn(i))
dt <- do.call('rbind',mylist)
dt <- as.data.table(dt)  
nas <- str_replace_all(nas,"[a-z]|[A-Z]|:|/|.txt|_|陈华珊老师课件|关于论文数据|wb","")
dt$names <- nas
dt <- rename(dt,(c(V1 = "text")))





#清理文本
##搜狗词典转存
#library(cidian)
#decode_scel(scel = "./14108.scel",cpp = TRUE)
#查看生成的词典文件
#讲生成的文件替换jiebaRD里面的用户词典文件
#scan(file="./14108.scel_2018-06-14_13_45_52.dict",
# what=character(),nlines=50,sep='\n',
#encoding='utf-8',fileEncoding='utf-8')
##清楚所有非中文字符,好像不加+?也是可以的
dt$text <- str_replace_all(dt$text,"[^\u4e00-\u9fa5]+?","")


##设置分词迭代器，itoken
##中文分词工具
##停止词文本
##在这一步已经分好词，并且去掉了相应的停止词了
jieba <- jiebaR::worker(stop_word = 'stop_word.txt')
tok_fun <- function(strings){ 
  lapply(strings, segment, jieba)}
dt1 <- itoken(dt$text, preprocessor = identity, tokenizer = tok_fun, ids = dt$names)

##试验错误,以下的两行代码是可以运行的
#dt1<- itoken(dt$text1,ids = dt$names)
#dt_vocal <- create_vocabulary(dt1)
##分词，create_vocabulary，英文里面直接分割即可，中文可就麻烦了，
##这里中文可不一样，官方案例是英文的，所以还需要自己处理一下。
##一些停用词、一些低频无效词都是文本噪声。
##所以针对停用词stopword可以在分词步骤create_vocabulary予以处理
##针对一些低频词的修剪，可以在分词create_vocabulary步骤之后以及设置、形成语料文件，
##vocab_vectorizer之前进行处理
dt_vocal <- create_vocabulary(dt1)
##设置、形成语料文件，vocab_vectorizer
vectorizer = vocab_vectorizer(dt_vocal)
##构建DTM矩阵，create_dtm
dtm_dt = create_dtm(dt1, vectorizer)
#检查id是否一致
identical(rownames(dtm_dt), dt$names) 
save(dtm_dt,file = "dtm.Rda")
colnames(dtm_dt)

##查看相应频数
library("slam")
fre_q <- col_sums(dtm_dt)
tail(fre_q,100)


##出现次数大于5
findFreqTerms(dtm_dt, 10)
##做下词云