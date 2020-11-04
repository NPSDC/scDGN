library(Rtsne)
library(umap)
library(ggpubr)

createPlot <- function(data, labels = NULL, batch = NULL, type ="PCA")
{
    if(!is.null(labels))
    {
        data <- data.frame(data)
        data <- cbind(data, labs = as.factor(labels), batch = as.factor(batch))
    }
    
    if(type == "PCA")
    {
        prData <- prcomp(data)
        if(!is.null(labels))
        {
            p1 <- autoplot(prData, data = data, color = "labs")
            p2 <- autoplot(prData, data = data, color = "batch")
            p <- ggarrange(p1, p2)
        }
        else
            p <- autoplot(prData)
    }
    else 
    {
        if(type == "tsne")
            redData <- as.data.frame(Rtsne(data[,1:(ncol(data)-2)])$Y)
        else
            redData <- as.data.frame(umap(data[,1:(ncol(data)-2)])$layout)
        colnames(redData) <- c("Dim1", "Dim2")
        redData <- cbind(redData, "labs" = data$labs, "batch" = data$batch)
        p <- ggplot(redData, aes(x=Dim1, y=Dim2))
        if(!is.null(labels))
        {
            p1 <- p + geom_point(aes(colour = labs))
            p2 <- p + geom_point(aes(colour = batch))
            p <- ggarrange(p1, p2)
        }
        else
            p <- p + geom_point()
    }
    gc()
    return(p)
}