library(ggplot2)

label_size <- 19
title_size <- 18
bar_width <- 0.7
border_width <- 0.5
annotation_size <- 6
color_map <- c("#c6e9f4", "#5679ba")
legend_position <- c(0.74, 0.915)
legend_font_size <- 17
title_size <- 20

# FOMC.
rephrase <- c(rep("Definition", 2), rep("Instruction", 2))
method <- rep(c("LLM-prompting", "LLM-Measure"), 2)
correlation <- c(0.842, 0.959, 0.768, 0.769)
data <- data.frame(rephrase, method, correlation)

data$method <- factor(data$method, levels = c("LLM-prompting", "LLM-Measure"))

p_fomc <- ggplot(data, aes(fill=method, y=correlation, x=rephrase)) + 
  geom_bar(width=bar_width, stat="identity", size=border_width, position=position_dodge()) +
  theme_bw() +
  theme(legend.position = legend_position, 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size),) +
  theme(legend.text=element_text(size = legend_font_size)) +
  geom_text(aes(label=sprintf("%0.3f", correlation)), vjust=-0.7, color="black",
            position = position_dodge(0.9), size=annotation_size) +
  xlab("") + 
  ylab("") +
  scale_fill_manual(values=color_map) +
  guides(fill=guide_legend(title="")) +
  theme(legend.title=element_text(size=0)) +
  coord_cartesian(ylim=c(0, 1)) +
  ggtitle("A. Fed's monetary policy stance") +
  theme(plot.title = element_text(size=title_size))

# Innovation.
rephrase <- c(rep("Definition", 2), rep("Instruction", 2))
method <- rep(c("LLM-prompting", "LLM-Measure"), 2)
correlation <- c(0.700, 0.906, 0.686, 0.713)
data <- data.frame(rephrase, method, correlation)
data$method <- factor(data$method, levels = c("LLM-prompting", "LLM-Measure"))

p_innovation <- ggplot(data, aes(fill=method, y=correlation, x=rephrase)) + 
  geom_bar(width=bar_width, stat="identity", size=border_width, position=position_dodge()) +
  theme_bw() +
  theme(legend.position = legend_position, 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size),) +
  theme(legend.text=element_text(size = legend_font_size)) +
  geom_text(aes(label=sprintf("%0.3f", correlation)), vjust=-0.7, color="black",
            position = position_dodge(0.9), size=annotation_size) +
  xlab("") + 
  ylab("") +
  scale_fill_manual(values=color_map) +
  guides(fill=guide_legend(title="")) +
  theme(legend.title=element_text(size=0)) +
  coord_cartesian(ylim=c(0, 1)) +
  ggtitle("B. Corporate innovation") +
  theme(plot.title = element_text(size=title_size))


# Information overload.
rephrase <- c(rep("Definition", 2), rep("Instruction", 2))
method <- rep(c("LLM-prompting", "LLM-Measure"), 2)
correlation <- c(0.539, 0.958, 0.488, 0.777)
data <- data.frame(rephrase, method, correlation)
data$method <- factor(data$method, levels = c("LLM-prompting", "LLM-Measure"))

p_complexity <- ggplot(data, aes(fill=method, y=correlation, x=rephrase)) + 
  geom_bar(width=bar_width, stat="identity", size=border_width, position=position_dodge()) +
  theme_bw() +
  theme(legend.position = legend_position, 
        axis.text = element_text(size=label_size),
        axis.title = element_text(size=title_size, face = "italic"),
        legend.title = element_text(size=title_size),
        legend.text = element_text(size=title_size),) +
  theme(legend.text=element_text(size = legend_font_size)) +
  geom_text(aes(label=sprintf("%0.3f", correlation)), vjust=-0.7, color="black",
            position = position_dodge(0.9), size=annotation_size) +
  xlab("") + 
  ylab("") +
  scale_fill_manual(values=color_map) +
  guides(fill=guide_legend(title="")) +
  theme(legend.title=element_text(size=0)) +
  coord_cartesian(ylim=c(0, 1)) +
  ggtitle("C. Information overload in reviews") +
  theme(plot.title = element_text(size=title_size))

g <- plot_grid(p_fomc, p_innovation, p_complexity, nrow = 1, ncol = 3)
y.grob <- textGrob("Pearson correlation coefficient", 
                   gp=gpar(fontface="bold", col="black", fontsize=23), rot=90)
x.grob <- textGrob("Sensitivity to different prompts", 
                   gp=gpar(fontface="bold", col="black", fontsize=23))
g <- grid.arrange(arrangeGrob(g, left = y.grob, bottom = x.grob))
ggsave("sensitivityBarCharts.png", g, height = 6, width = 16, dpi = 900)


