##加载包
library("chinese.misc")
library("data.table")
library(stringr)
library(reshape)
library(jiebaR)
library(text2vec)
setwd("E:\\shujufenxi\\R_workplace\\chs")

##批量读取，如果路径有中文的话，提取变量名消除其中的路径中文词
files <- dir_or_file("E:\\shujufenxi\\R_workplace\\chs\\wb", special='txt$')
mylist <- lapply(files,function(i) scancn(i))
dt <- do.call('rbind',mylist)
dt <- as.data.table(dt)  
#提取文件名,并加进数据中
nas <- files 
nas <- str_replace_all(nas,"[a-z]|[A-Z]|:|/|.txt|_","")
dt$names <- nas
##这里数据的text的变量是V1，且与names的顺序是，因此要对其进行处理
##添加一个新列，再把原来的列去掉，顺序就好了
dt$text <- dt$V1
dt <- dt[,-1]
save(dt,file = "gzdt.Rda")
##加载数据
load("gzdt.Rda")




##搜狗词典转存
#library(cidian)
#decode_scel(scel = "./14108.scel",cpp = TRUE)
#查看生成的词典文件
#讲生成的文件替换jiebaRD里面的用户词典文件
#scan(file="./14108.scel_2018-06-14_13_45_52.dict",
         # what=character(),nlines=50,sep='\n',
          #encoding='utf-8',fileEncoding='utf-8')


#清理文本
##清楚所有非中文字符,好像不加+?也是可以的
dt$text <- str_replace_all(dt$text,"[^\u4e00-\u9fa5]+?","")

##设置分词迭代器，itoken
##中文分词工具
##停止词文本
##在这一步已经分好词，并且去掉了相应的停止词了
##itoken是设置好参数，才creat一步才会具体的进行分词的步骤
##之所以要这么多步骤是应为效率很快
jieba<-jiebaR::worker(stop_word = 'stop_word.txt')
tok_fun <-function(strings){ 
  lapply(strings, segment, jieba)}
dt1 <- itoken(dt$text, preprocessor = identity, tokenizer = tok_fun, ids = dt$names)
dt_vocal <- create_vocabulary(dt1,ngram = c(1L, 2L))
##针对一些低频词的修剪，可以在分词create_vocabulary步骤之后以及设置、形成语料文件
##设置低频界限为5
pruned_vocab = prune_vocabulary(dt_vocal, term_count_min = 5 )
##设置、形成语料文件，vocab_vectorizer
vectorizer = vocab_vectorizer(pruned_vocab)
##构建DTM矩阵，create_dtm
dtm_dt = create_dtm(dt1, vectorizer)
#检查id是否一致
identical(rownames(dtm_dt), dt$names) 
##save(dtm_dt,file = "dtm.Rda")
colnames(dtm_dt)
##查看相应频数
library(slam)
a <- col_sums(dtm_dt)
##出现次数大于5
tail(a,500)
t <- apply(dtm_dt,1,function(y)(names(sort(y,decreasing = T)[1:10])))
##查看每一个工种好对应的高频词
t1 <- apply(dtm_dt,1,function(y)(names(sort(y,decreasing = T)[1:10])))
m2 <- apply(dtm_dt,1,function(x)rbind(names(sort(x,decreasing=T)[1:10]),sort(x,decreasing=T)[1:10]))
##做下词云

##每一行词频在前几的词语

##主题模型
##参数其中α，大家可以调大调小了试试看，调大了的结果是每个文档接近同一个topic，
##即让p(wi|topici)发挥的作用小，这样p(di|topici)发挥的作用就大。其中的β，
##调大的结果是让p(di|topici)发挥的作用变下，而让p(wi|topici)发挥的作用变大，
##体现在每个topic更集中在几个词汇上面，或者而每个词汇都尽可能的百分百概率转移到一个topic上
set.seed(888)
nt <- 6
lda_model = LDA$new(n_topics = nt,doc_topic_prior = 50 / nt, topic_word_prior = 1 /nt )
#model1生成的是一个矩阵值是对应每个公众号在某一主题上对应的比例
tomodel1 <- lda_model$fit_transform(dtm_dt,n_iter = 2000)
tomodel2 <- lda_model$transform(dtm_dt,n_iter = 2000)
top_word <- lda_model$get_top_words(n = 20, topic_number = 1L:6L, lambda = 1)
##主题词语矩阵
topicsterms_ldatext2vec <- lda_model$get_top_words(n = nrow(pruned_vocab), topic_number = 1L:6L,
                                                   lambda = 1)

##主题模型可视化
library(LDAvis)
lda_model$plot(lambda.step = 0.1, reorder.topics = FALSE)

##感觉这里应该还有各个公众号在主题上的分布的情况

#网络
#直接双模转换成单模来做，但是好像这样的方法对于有权网来说不是而别的好
#陈老师的高斯模型的文章有讲到这个事情
#领接矩阵
#netwe <- tomodel1 %*% t(tomodel1)
#diag(netwe) <- 0
##矩阵二值化，暂时设置成0.2吧
bar <- function(y){
  ifelse(y[] > 0.2, 1, 0)
}
#将单模矩阵二值化
##
netwe <- apply(tomodel1, 1, bar)
#转成双模矩阵
#netwe1 <- t(netwe)%*% netwe

##生成igraht格式文件
library(igraph)
#netwe_ig <-  graph.adjacency(netwe>0, mode="undirected",diag = FALSE)
#plot(netwe_ig, vertex.color = 'violet', vertex.size = 20, edge.color = 'yellow')

##用高斯模型来生成网络
library(glasso)
#View(tomodel1)
#la_td <- t(tomodel1)
result = glasso(cov(netwe), rho = 0.1, penalize.diagonal=F)
wi_to_pcor <- function(wi) {
  p = -wi
  d = 1/sqrt(diag(wi))
  pcor = diag(d)%*%p%*%diag(d)
  diag(pcor) = 0
  colnames(pcor) <- seq_len(ncol(pcor))
  pcor
}

pcor = wi_to_pcor(result$wi)
g = graph.adjacency(pcor > 0, mode="undirected", diag = FALSE)
pdf(encoding = "GB1")
par(mar=c(0,0,2,0))
plot(g, vertex.color = 'white', edge.color = 'black', vertex.label = dt$names,
     main = "公众号的主题概率分布联系图")


###下面的代码不用了，就用惩罚项是0.1
rhos <- c(0.05, 0.08, 0.1, 0.2, 0.8)
ret <- lapply(rhos, glasso, s = cov(netwe), penalize.diagonal = F)

wi <- lapply(ret, getElement, 'wi')
ret.pcor <- lapply(wi, wi_to_pcor)
ret.mat <- lapply(ret.pcor, function(x) x > 0)
ret.g <- lapply(ret.mat, graph.adjacency, mode="undirected", diag = FALSE)

for (i in seq_along(ret.g)) {
  plot(ret.g[[i]], vertex.color = 'white', vertex.size = 20, edge.color = 'black', main = sprintf("rho = %.2f", rhos[i]))
}






