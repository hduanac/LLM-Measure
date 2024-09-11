library(ggplot2)
library(gridExtra)
library(cowplot)
library(scales)
library(lsa)

load("complexity_rephrase.RData")
load("complexity_regression.RData")
load("fomc_rephrase.RData")
load("fomc_regression.RData")
load("innovation_rephrase.RData")
load("innovation_regression.RData")

complexity_rephrase[, 6:8] <- scale(complexity_rephrase[, 6:8])
fomc_rephrase[, 1:8] <- scale(fomc_rephrase[, 1:8])
innovation_rephrase[, 1:8] <- scale(innovation_rephrase[, 1:8])

dot_color_complexity <- "#3a9295"
dot_color_fomc <- "#3a9295"
dot_color_innovation <- "#3a9295"
dot_size <- 4
dot_shape_complexity <- 0
dot_shape_fomc <- 1
dot_shape_innovation <- 2
se_color <- 'azure4'
label_size <- 16
line_color <- "black"
offset <- 0
ann_size <- 5
x_acc <- 0.1
y_acc <- 0.1
hjust <- 0

correlation_method <- "pearson"
corr <- cor.test(scale(complexity_rephrase$original), scale(complexity_rephrase$original_128), method=correlation_method)$estimate

complexity_llmaai <- ggplot(complexity_rephrase, aes(x = original, y = original_128)) + 
  geom_point(color = dot_color_complexity, size = dot_size, shape = dot_shape_complexity) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("C") +
  annotate("text", 
           min(complexity_rephrase$original) + offset, 
           max(complexity_rephrase$original_128) - 1.2, 
           label = sprintf("r:%0.4f", corr), size=ann_size, parse = TRUE, hjust=hjust) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


corr <- cor.test(scale(fomc_rephrase$original), scale(fomc_rephrase$original_128), method=correlation_method)$estimate

fomc_llmaai <- ggplot(fomc_rephrase, aes(x = original, y = original_128)) + 
  geom_point(color = dot_color_fomc, size = dot_size, shape = dot_shape_fomc) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("A") +
  annotate("text", 
           min(fomc_rephrase$original) + offset, 
           max(fomc_rephrase$original_128) - 0.36, 
           label = sprintf("r:%0.4f", corr), size=ann_size, parse = TRUE, hjust=hjust) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))



corr <- cor.test(scale(innovation_rephrase$original), scale(innovation_rephrase$original_128), method=correlation_method)$estimate

innovation_llmaai <- ggplot(innovation_rephrase, aes(x = original, y = original_128)) + 
  geom_point(color = dot_color_innovation, size = dot_size, shape = dot_shape_innovation) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("B") +
  annotate("text", 
           min(innovation_rephrase$original) + offset, 
           max(innovation_rephrase$original_128) - 0.9, 
           label = sprintf("r:%0.4f", corr), size=ann_size, parse = TRUE, hjust=hjust) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


g <- plot_grid(
          fomc_llmaai, 
          innovation_llmaai,
          complexity_llmaai,
          nrow = 1, ncol = 3)

y.grob <- textGrob("Concept value (128 samples)", 
                   gp=gpar(fontface="bold", col="black", fontsize=13), rot=90)
x.grob <- textGrob("Concept value (64 samples)", 
                   gp=gpar(fontface="bold", col="black", fontsize=13))
g <- grid.arrange(arrangeGrob(g, left = y.grob, bottom = x.grob))
ggsave("probingSize.png", g, height = 3, width = 9,  dpi = 900)
