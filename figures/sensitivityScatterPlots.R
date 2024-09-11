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

dot_color_llmaai <- "firebrick4"
dot_color_prompt <- "blue4"
dot_size <- 4
dot_shape_llammi <- 1
dot_shape_prompt <- 4
se_color <- 'azure4'
label_size <- 16
line_color <- "black"
offset <- 0
ann_size <- 5
x_acc <- 0.1
y_acc <- 0.1

# Consumer review.
correlation_method <- "pearson"
corr <- cor.test(scale(complexity_rephrase$original), scale(complexity_rephrase$rephrase_def_64), method=correlation_method)$estimate

complexity_llmaai_def <- ggplot(complexity_rephrase, aes(x = original, y = rephrase_def_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("I") +
  annotate("text", 
           min(complexity_rephrase$original) + offset, 
           max(complexity_rephrase$rephrase_def_64), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


corr <- cor.test(scale(complexity_rephrase$output_score_ori), scale(complexity_rephrase$output_score_rephrase_def), method=correlation_method)$estimate

complexity_prompt_def <- ggplot(complexity_rephrase, aes(x = output_score_ori, y = output_score_rephrase_def)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("J") +
  annotate("text", 
           min(complexity_rephrase$output_score_ori) + offset, 
           max(complexity_rephrase$output_score_rephrase_def), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


corr <- cor.test(scale(complexity_rephrase$original), scale(complexity_rephrase$rephrase_inst_64), method=correlation_method)$estimate

complexity_llmaai_inst <- ggplot(complexity_rephrase, aes(x = original, y = rephrase_inst_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("K") +
  annotate("text", 
           min(complexity_rephrase$original) + offset, 
           max(complexity_rephrase$rephrase_inst_64), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))



corr <- cor.test(scale(complexity_rephrase$output_score_ori), scale(complexity_rephrase$output_score_rephrase_inst), method=correlation_method)$estimate

complexity_prompt_inst <- ggplot(complexity_rephrase, aes(x = output_score_ori, y = output_score_rephrase_inst)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("L") +
  annotate("text", 
           min(complexity_rephrase$output_score_ori) + offset, 
           max(complexity_rephrase$output_score_rephrase_inst), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


# FOMC.
corr <- cor.test(scale(fomc_rephrase$original), scale(fomc_rephrase$rephrase_def_64), method=correlation_method)$estimate

fomc_llmaai_def <- ggplot(fomc_rephrase, aes(x = original, y = rephrase_def_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("A") +
  annotate("text", 
           min(fomc_rephrase$original) + offset, 
           max(fomc_rephrase$rephrase_def_64), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))



corr <- cor.test(scale(fomc_rephrase$output_score_ori), scale(fomc_rephrase$output_score_rephrase_def), method=correlation_method)$estimate

fomc_prompt_def <- ggplot(fomc_rephrase, aes(x = output_score_ori, y = output_score_rephrase_def)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("B") +
  annotate("text", 
           min(fomc_rephrase$output_score_ori) + offset, 
           max(fomc_rephrase$output_score_rephrase_def), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))

  


corr <- cor.test(scale(fomc_rephrase$original), scale(fomc_rephrase$rephrase_inst_64), method=correlation_method)$estimate

fomc_llmaai_inst <- ggplot(fomc_rephrase, aes(x = original, y = rephrase_inst_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("C") +
  annotate("text", 
           min(fomc_rephrase$original) + offset, 
           max(fomc_rephrase$rephrase_inst_64), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))

  
  
  
corr <- cor.test(scale(fomc_rephrase$output_score_ori), scale(fomc_rephrase$output_score_rephrase_inst), method=correlation_method)$estimate

fomc_prompt_inst <- ggplot(fomc_rephrase, aes(x = output_score_ori, y = output_score_rephrase_inst)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("D") +
  annotate("text", 
           min(fomc_rephrase$output_score_ori) + offset, 
           max(fomc_rephrase$output_score_rephrase_inst), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


# Innovation.
corr <- cor.test(scale(innovation_rephrase$original), scale(innovation_rephrase$rephrase_def_64), method=correlation_method)$estimate

innovation_llmaai_def <- ggplot(innovation_rephrase, aes(x = original, y = rephrase_def_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("E") +
  annotate("text", 
           min(innovation_rephrase$original) + offset, 
           max(innovation_rephrase$rephrase_def_64), 
           label = sprintf("r:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))



corr <- cor.test(scale(innovation_rephrase$output_score_ori), scale(innovation_rephrase$output_score_rephrase_def), method=correlation_method)$estimate

innovation_prompt_def <- ggplot(innovation_rephrase, aes(x = output_score_ori, y = output_score_rephrase_def)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("F") +
  annotate("text", 
           min(innovation_rephrase$output_score_ori) + offset, 
           max(innovation_rephrase$output_score_rephrase_def), 
           label = sprintf("rho:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))




corr <- cor.test(scale(innovation_rephrase$original), scale(innovation_rephrase$rephrase_inst_64), method=correlation_method)$estimate

innovation_llmaai_inst <- ggplot(innovation_rephrase, aes(x = original, y = rephrase_inst_64)) + 
  geom_point(color = dot_color_llmaai, size = dot_size, shape = dot_shape_llammi) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("G") +
  annotate("text", 
           min(innovation_rephrase$original) + offset, 
           max(innovation_rephrase$rephrase_inst_64), 
           label = sprintf("rho:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))




corr <- cor.test(scale(innovation_rephrase$output_score_ori), scale(innovation_rephrase$output_score_rephrase_inst), method=correlation_method)$estimate

innovation_prompt_inst <- ggplot(innovation_rephrase, aes(x = output_score_ori, y = output_score_rephrase_inst)) + 
  geom_point(color = dot_color_prompt, size = dot_size, shape = dot_shape_prompt) + 
  geom_smooth(method = lm, color=line_color, fill=se_color) +
  theme_bw() +
  theme(text = element_text(size = label_size)) + 
  theme(axis.title.x=element_blank(),
        axis.title.y=element_blank(),) +
  ggtitle("H") +
  annotate("text", 
           min(innovation_rephrase$output_score_ori) + offset, 
           max(innovation_rephrase$output_score_rephrase_inst), 
           label = sprintf("rho:%0.3f", corr), size=ann_size, parse = TRUE, hjust=0) +
  scale_x_continuous(labels = number_format(accuracy = x_acc)) +
  scale_y_continuous(labels = number_format(accuracy = y_acc))


g <- plot_grid(
          fomc_llmaai_def, 
          fomc_prompt_def, 
          fomc_llmaai_inst,
          fomc_prompt_inst,
          innovation_llmaai_def, 
          innovation_prompt_def, 
          innovation_llmaai_inst,
          innovation_prompt_inst,
          complexity_llmaai_def, 
          complexity_prompt_def, 
          complexity_llmaai_inst,
          complexity_prompt_inst,
          nrow = 3, ncol = 4)

y.grob <- textGrob("Concept value (rephrased prompts)", 
                   gp=gpar(fontface="bold", col="black", fontsize=18), rot=90)
x.grob <- textGrob("Concept value (original prompts)", 
                   gp=gpar(fontface="bold", col="black", fontsize=18))
g <- grid.arrange(arrangeGrob(g, left = y.grob, bottom = x.grob))
ggsave("sensitivityScatterPlots.png", g, height = 9, width = 12,  dpi = 900)



