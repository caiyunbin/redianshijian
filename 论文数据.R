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
#看最后200个的词语是什么
fre_q <- colnames(dtm_dt)
sorted <- dt_vocal[order(dt_vocal[,2],decreasing = T),]
#词云制作
library(wordcloud2)
wordcloud2(data = sorted)

#取出公众号每一行前10个词
t <- apply(dtm_dt,1,function(y)(names(sort(y,decreasing = T)[1:10])))
#取出公众号每一行前10和那个名字
m <- apply(dtm_dt,1,function(x)rbind(names(sort(x,decreasing=T)[1:10]),sort(x,decreasing=T)[1:10]))


#设置分词最小概率，出现5次以下的不考虑
pruned_vocab = prune_vocabulary(dt_vocal, term_count_min = 5 )
##主题模型
##参数其中α，大家可以调大调小了试试看，调大了的结果是每个文档接近同一个topic，
##即让p(wi|topici)发挥的作用小，这样p(di|topici)发挥的作用就大。其中的β，
##调大的结果是让p(di|topici)发挥的作用变下，而让p(wi|topici)发挥的作用变大，
##体现在每个topic更集中在几个词汇上面，或者而每个词汇都尽可能的百分百概率转移到一个topic上




set.seed(888)
nt <- 6
lda_model = LDA$new(n_topics = nt,doc_topic_prior = 50 / nt, topic_word_prior = 1 /nt )
#model1生成的是一个矩阵值是对应每个公众号在某一主题上对应的比例,抽样迭代的次数为2000
tomodel1 <- lda_model$fit_transform(dtm_dt,n_iter = 2000)
tomodel2 <- lda_model$transform(dtm_dt,n_iter = 2000)
top_word <- lda_model$get_top_words(n = 20, topic_number = 1L:6L, lambda = 1)
##主题词语矩阵，最高的相关系数取1.
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
result = glasso(cov(netwe), rho = 0.1, penalize.diagonal=T)
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


#词向量的glove模型
#将整个词典进行向量化操作并赋值
vec <- vocab_vectorizer(dt_vocal)
# 考虑词的前后5个词，构建一个词频共现矩阵
tcm <- create_tcm(dt1, vec, skip_grams_window = 10L)

#词汇表的最大长度为40
glove = GlobalVectors$new(word_vectors_size = 40, vocabulary = dt_vocal, x_max = 10)

#进行glove模型拟合，放入词共现矩阵，相关系数阈值设定为0.01，之下的不考虑
wv_main = glove$fit_transform(tcm, n_iter = 10, convergence_tol = 0.01)
dim(wv_main)
wv_context = glove$components
dim(wv_context)
wv_main[1,1]
t(wv_context)[1,1]
word_vectors = wv_main + t(wv_context)

#构造“A+B”向量
relation = word_vectors["肌肤", , drop = FALSE] +
  word_vectors["产品", , drop = FALSE]

#计算相关性，使用角度，查看相关性最高的词，优化方法，标准化，加入惩罚项，使用"l2"范数
cos_sim = sim2(x = word_vectors, y = relation, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 20)


#构造“中国+美国”向量
relation = word_vectors["爱", , drop = FALSE] 

relation = word_vectors["孩子","电影", drop = FALSE] +
  word_vectors["孩子", , drop = FALSE]
cos_sim = sim2(x = word_vectors, y = relation, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 20)



#计算相关性，查看相关性最高的词 
cos_sim = sim2(x = word_vectors, y = relation, method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 20)




