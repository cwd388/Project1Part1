#Project 1 - Part I
#Chad W. Dunham

#Naive Bayes Spam Classifier - This algorithm looks for words that are 
#noticeably more likely to occur in spam messages or noticeably more likely to 
#occur in ham messages for many words.  To do this, the algorithm "[computes] 
#the probability of seeing the exact contents of an email conditioned on the 
#email being assumed to be spam, and . . . the probability of seeing the same 
#email's contents conditioned on the email being assumed to be ham.



#Importing needed packages
#install.packages("tm", repos="http://cran.mtu.edu/")
library(tm)
library(ggplot2)


#Setting paths for files
spam.path = "data/spam/"
spam2.path = "data/spam_2/"
easyham.path = "data/easy_ham/"
easyham2.path = "data/easy_ham_2/"
hardham.path = "data/hard_ham/"
hardham2.path = "data/hard_ham_2/"


# Creating the "motivating plot"... note this is not really part of the 
#classifier but is included in the author's code.
x <- runif(1000, 0, 40)
y1 <- cbind(runif(100, 0, 10), 1)
y2 <- cbind(runif(800, 10, 30), 2)
y3 <- cbind(runif(100, 30, 40), 1)

val <- data.frame(cbind(x, rbind(y1, y2, y3)),
                  stringsAsFactors = TRUE)

ex1 <- ggplot(val, aes(x, V2)) +
  geom_jitter(aes(shape = as.factor(V3)),
              position = position_jitter(height = 2)) +
  scale_shape_discrete(guide = "none", solid = FALSE) +
  geom_hline(aes(yintercept = c(10,30)), linetype = 2) +
  theme_bw() +
  xlab("X") +
  ylab("Y")
ggsave(plot = ex1,
       filename = file.path("images", "00_Ex1.pdf"),
       height = 10,
       width = 10)

#Text-mining function
get.msg = function(path) {
  con = file(path, open="rt", encoding="native.enc") 
  #This is changed to correspond with the errata fixes in the book
  text = readLines(con)
  msg = text[seq(which(text=="")[1]+1, length(text), 1)]
  #Above edited to include update code from authors
  close(con)
  return(paste(msg, collapse="\n"))
}


#Retrieving the messages from all of the spam and ham emails
spam.docs = dir(spam.path)
spam.docs = spam.docs[which(spam.docs!="cmds")]
all.spam = sapply(spam.docs, function(p) get.msg(paste(spam.path, p, sep="")))



#Creating the Term Document Matrix
get.tdm = function(doc.vec)  {
  doc.corpus = Corpus(VectorSource(doc.vec))
  control = list(stopwords=TRUE, removePunctuation=TRUE, removeNumbers=TRUE, minDocFreq=2)
  doc.dtm = TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}
spam.tdm = get.tdm(all.spam)


#Building TDM training data
spam.matrix = as.matrix(spam.tdm)
spam.counts = rowSums(spam.matrix)
spam.df = data.frame(cbind(names(spam.counts), as.numeric(spam.counts)), stringsAsFactors=FALSE)
names(spam.df) = c("term", "frequency")
spam.df$frequency = as.numeric(spam.df$frequency)

#Generating training data
spam.occurrence = sapply(1:nrow(spam.matrix), function(i) {length(which(spam.matrix[i,] > 0))/ncol(spam.matrix)})
spam.density = spam.df$frequency/sum(spam.df$frequency)

#Adding the training data to the original data frame
spam.df = transform(spam.df, density=spam.density, occurrence=spam.occurrence)

head(spam.df[with(spam.df, order(-occurrence)),]) #Summary printout from book



#REDOING ABOVE FOR EASY HAM
#Retrieving the messages from all of the spam and ham emails
easyham.docs = dir(easyham.path)
easyham.docs = easyham.docs[which(easyham.docs!="cmds")]
all.easyham = sapply(easyham.docs, function(p) get.msg(paste(easyham.path, p, sep="")))
#Running the Term Document Matrix (using previously created function)
easyham.tdm = get.tdm(all.easyham)
#Building TDM training data
easyham.matrix = as.matrix(easyham.tdm)
easyham.counts = rowSums(easyham.matrix)
easyham.df = data.frame(cbind(names(easyham.counts), as.numeric(easyham.counts)), stringsAsFactors=FALSE)
names(easyham.df) = c("term", "frequency")
easyham.df$frequency = as.numeric(easyham.df$frequency)
#Generating training data
easyham.occurrence = sapply(1:nrow(easyham.matrix), function(i) {length(which(easyham.matrix[i,] > 0))/ncol(easyham.matrix)})
easyham.density = easyham.df$frequency/sum(easyham.df$frequency)
#Adding the training data to the original data frame
easyham.df = transform(easyham.df, density=easyham.density, occurrence=easyham.occurrence)

head(easyham.df[with(easyham.df, order(-occurrence)),]) #Summary printout from book



#Setting up the email classifier function
classify.email = function(path, training.df, prior=.5, c=1e-6)  {
  msg = get.msg(path)
  msg.tdm = get.tdm(msg)
  msg.freq = rowSums(as.matrix(msg.tdm))
  msg.match = intersect(names(msg.freq), training.df$term)
  if(length(msg.match) < 1) {
    return(prior*c^(length(msg.freq)))
  }
  else {
    match.probs = training.df$occurrence[match(msg.match, training.df$term)]
    return(prior * prod(match.probs) * c^(length(msg.freq)-length(msg.match)))
  }
}


#Testing the classifier on hard ham
hardham.docs = dir(hardham.path)
hardham.docs = hardham.docs[which(hardham.docs != "cmds")]

hardham.spamtest = sapply(hardham.docs, function(p) classify.email(paste(hardham.path, p, sep="/"), training.df = spam.df))

hardham.hamtest = sapply(hardham.docs, function(p) classify.email(paste(hardham.path, p, sep="/"), training.df = easyham.df))

hardham.res = ifelse(hardham.spamtest > hardham.hamtest, FALSE, TRUE)
summary(hardham.res)



#Testing the classifier against all types of emails
spam.classifier = function(path)  {
  pr.spam = classify.email(path, spam.df)
  pr.ham = classify.email(path, easyham.df)
  return(c(pr.spam, pr.ham, ifelse(pr.spam > pr.ham, 1, 0)))
}

# Getting the lists of all the email messages
easyham2.docs <- dir(easyham2.path)
easyham2.docs <- easyham2.docs[which(easyham2.docs != "cmds")]

hardham2.docs <- dir(hardham2.path)
hardham2.docs <- hardham2.docs[which(hardham2.docs != "cmds")]

spam2.docs <- dir(spam2.path)
spam2.docs <- spam2.docs[which(spam2.docs != "cmds")]


# Classifying all of the classes of emails
easyham2.class <- suppressWarnings(lapply(easyham2.docs,
                                          function(p)
                                          {
                                            spam.classifier(file.path(easyham2.path, p))
                                          }))
hardham2.class <- suppressWarnings(lapply(hardham2.docs,
                                          function(p)
                                          {
                                            spam.classifier(file.path(hardham2.path, p))
                                          }))
spam2.class <- suppressWarnings(lapply(spam2.docs,
                                       function(p)
                                       {
                                         spam.classifier(file.path(spam2.path, p))
                                       }))

# Create a single, final, data frame with all of the classification data in it
easyham2.matrix <- do.call(rbind, easyham2.class)
easyham2.final <- cbind(easyham2.matrix, "EASYHAM")

hardham2.matrix <- do.call(rbind, hardham2.class)
hardham2.final <- cbind(hardham2.matrix, "HARDHAM")

spam2.matrix <- do.call(rbind, spam2.class)
spam2.final <- cbind(spam2.matrix, "SPAM")

class.matrix <- rbind(easyham2.final, hardham2.final, spam2.final)
class.df <- data.frame(class.matrix, stringsAsFactors = FALSE)
names(class.df) <- c("Pr.SPAM" ,"Pr.HAM", "Class", "Type")
class.df$Pr.SPAM <- as.numeric(class.df$Pr.SPAM)
class.df$Pr.HAM <- as.numeric(class.df$Pr.HAM)
class.df$Class <- as.logical(as.numeric(class.df$Class))
class.df$Type <- as.factor(class.df$Type)

head(class.df)

#Matrix of classifier results

# Creating a final plot of results
class.plot <- ggplot(class.df, aes(x = log(Pr.HAM), log(Pr.SPAM))) +
  geom_point(aes(shape = Type, alpha = 0.5)) +
  stat_abline(yintercept = 0, slope = 1) +
  scale_shape_manual(values = c("EASYHAM" = 1,
                                "HARDHAM" = 2,
                                "SPAM" = 3),
                     name = "Email Type") +
  scale_alpha(guide = "none") +
  xlab("log[Pr(HAM)]") +
  ylab("log[Pr(SPAM)]") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.text.y = element_blank())
ggsave(plot = class.plot, filename = file.path("images", "03_final_classification.pdf"),
       height = 10,
       width = 10)

get.results <- function(bool.vector)
{
  results <- c(length(bool.vector[which(bool.vector == FALSE)]) / length(bool.vector),
               length(bool.vector[which(bool.vector == TRUE)]) / length(bool.vector))
  return(results)
}

# Saving the results as a 2x3 table
easyham2.col <- get.results(subset(class.df, Type == "EASYHAM")$Class)
hardham2.col <- get.results(subset(class.df, Type == "HARDHAM")$Class)
spam2.col <- get.results(subset(class.df, Type == "SPAM")$Class)

class.res <- rbind(easyham2.col, hardham2.col, spam2.col)
colnames(class.res) <- c("NOT SPAM", "SPAM")
print(class.res)

#END OF CODE